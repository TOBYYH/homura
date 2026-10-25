//! A restricted unpickler: the part of Python's pickle format that PyTorch checkpoints use, read
//! without executing any code.
//!
//! Parsing builds an object graph in an arena: every opcode creates at most one node, and the
//! stack and memo hold node ids, so a memo reference is the same object, as in Python, and the
//! arena grows with the input. [`Pickle::to_value`] then turns the graph into a [`Value`] tree,
//! where an object referenced several times is copied each time. That can expand a small input
//! enormously, so it is charged against budgets that grow with the input's size, before anything
//! is allocated, and cycles are an error.
//!
//! Only `collections.OrderedDict`, PyTorch's tensor rebuild functions and its storage classes
//! can be referenced, and only the first two called. Tensors are rebuilt by a [`Tensors`]
//! implementation from their call's arguments.

use std::collections::HashMap;
use std::rc::Rc;

use super::value::{Dict, Key, Value};
use crate::nn::ModuleVersions;
use crate::{DType, Error, Result, Tensor};

/// The id of a node in the arena.
pub(super) type Id = u32;

/// A global that a pickle can reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Global {
    OrderedDict,
    /// `torch._utils._rebuild_tensor_v2`.
    RebuildTensorV2,
    /// `torch._utils._rebuild_parameter`.
    RebuildParameter,
    /// `torch.FloatStorage` and the others: the class of a storage.
    Storage(DType),
}

const STORAGES: [(&str, DType); 10] = [
    ("FloatStorage", DType::F32),
    ("DoubleStorage", DType::F64),
    ("HalfStorage", DType::F16),
    ("BFloat16Storage", DType::BF16),
    ("LongStorage", DType::I64),
    ("IntStorage", DType::I32),
    ("ShortStorage", DType::I16),
    ("CharStorage", DType::I8),
    ("ByteStorage", DType::U8),
    ("BoolStorage", DType::Bool),
];

impl Global {
    fn find(module: &str, name: &str) -> Result<Global> {
        let global = match (module, name) {
            ("collections", "OrderedDict") => Some(Global::OrderedDict),
            ("torch._utils", "_rebuild_tensor_v2") => Some(Global::RebuildTensorV2),
            ("torch._utils", "_rebuild_parameter") => Some(Global::RebuildParameter),
            ("torch", _) => STORAGES
                .iter()
                .find(|(n, _)| *n == name)
                .map(|&(_, dtype)| Global::Storage(dtype)),
            _ => None,
        };
        global.ok_or_else(|| {
            Error::Format(format!(
                "unsupported pickle data: the global {module}.{name} is not supported (only \
                 state dicts and checkpoints of tensors, numbers, strings, lists, tuples and \
                 dicts are)"
            ))
        })
    }

    pub(super) fn name(self) -> String {
        match self {
            Global::OrderedDict => "collections.OrderedDict".into(),
            Global::RebuildTensorV2 => "torch._utils._rebuild_tensor_v2".into(),
            Global::RebuildParameter => "torch._utils._rebuild_parameter".into(),
            Global::Storage(dtype) => {
                let name = STORAGES.iter().find(|(_, d)| *d == dtype).unwrap().0;
                format!("torch.{name}")
            }
        }
    }
}

/// A dict key: strings are interned, so equal strings have the same id.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum KeyId {
    Int(i64),
    Str(Id),
}

/// A `dict` or `OrderedDict` in the arena.
#[derive(Debug, Default)]
pub(super) struct DictNode {
    pub(super) ordered: bool,
    pub(super) items: Vec<(KeyId, Id)>,
    index: HashMap<KeyId, usize>,
    /// The `_metadata` attribute that `BUILD` bound on an `OrderedDict`. Like
    /// `inst.__dict__.update(state)`, each `BUILD` rebinds only the attributes its state has; the
    /// bound object is shared, so later changes to it are seen.
    pub(super) metadata: Option<Id>,
}

/// An object in the arena.
#[derive(Debug)]
pub(super) enum Node {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    Tuple(Vec<Id>),
    List(Vec<Id>),
    Dict(DictNode),
    Global(Global),
    /// `func(*args)` for a function that rebuilds a tensor; `args` is a tuple.
    Call {
        func: Global,
        args: Id,
    },
    /// A persistent id (`BINPERSID`): a storage, in PyTorch's files.
    Persistent(Id),
}

impl Node {
    pub(super) fn type_name(&self) -> &'static str {
        match self {
            Node::None => "None",
            Node::Bool(_) => "bool",
            Node::Int(_) => "int",
            Node::Float(_) => "float",
            Node::Str(_) => "str",
            Node::Tuple(_) => "tuple",
            Node::List(_) => "list",
            Node::Dict(d) if d.ordered => "OrderedDict",
            Node::Dict(_) => "dict",
            Node::Global(_) => "global",
            Node::Call { .. } => "tensor",
            Node::Persistent(_) => "persistent id",
        }
    }
}

/// A parsed pickle: its object graph and the object it returns.
#[derive(Debug)]
pub(super) struct Pickle {
    nodes: Vec<Node>,
    root: Id,
    /// The size of the input, which the budgets of [`to_value`](Pickle::to_value) grow with.
    input_len: usize,
}

/// Rebuilds tensors for [`Pickle::to_value`].
pub(super) trait Tensors {
    /// The tensor that the call node `call`, `func(*args)`, returns, where `args` is a tuple
    /// node of `pickle`. [`Pickle::to_value`] asks again each time the node is referenced.
    fn tensor(&mut self, pickle: &Pickle, call: Id, func: Global, args: Id) -> Result<Tensor>;
}

/// The most entries the memo can have. Each one takes an opcode, so the input's size bounds
/// them too; this only keeps the map small for inputs of hundreds of megabytes.
const MEMO_LIMIT: usize = 1 << 22;

/// The deepest nesting of containers that [`Pickle::to_value`] accepts.
const MAX_DEPTH: usize = 256;

/// The budgets of [`Pickle::to_value`], for an input of `len` bytes: the number of values it
/// creates (each container's items, a tensor or a version entry counts as one) and the total
/// bytes of strings it copies. Without shared references, an input creates at most one value
/// per byte and copies each byte of its strings once, so the budgets leave 8 times as much for
/// objects that are referenced several times.
fn budgets(len: usize) -> (u64, u64) {
    let len = len as u64;
    (65_536 + 8 * len, (1 << 20) + 8 * len)
}

fn bad(msg: &str) -> Error {
    Error::Format(format!("invalid pickle data: {msg}"))
}

fn unsupported(msg: &str) -> Error {
    Error::Format(format!("unsupported pickle data: {msg}"))
}

mod op {
    pub const PROTO: u8 = 0x80;
    pub const FRAME: u8 = 0x95;
    pub const STOP: u8 = b'.';
    pub const MARK: u8 = b'(';
    pub const POP: u8 = b'0';
    pub const POP_MARK: u8 = b'1';
    pub const MEMOIZE: u8 = 0x94;
    pub const BINPUT: u8 = b'q';
    pub const LONG_BINPUT: u8 = b'r';
    pub const BINGET: u8 = b'h';
    pub const LONG_BINGET: u8 = b'j';
    pub const NONE: u8 = b'N';
    pub const NEWTRUE: u8 = 0x88;
    pub const NEWFALSE: u8 = 0x89;
    pub const BININT: u8 = b'J';
    pub const BININT1: u8 = b'K';
    pub const BININT2: u8 = b'M';
    pub const LONG1: u8 = 0x8a;
    pub const BINFLOAT: u8 = b'G';
    pub const BINUNICODE: u8 = b'X';
    pub const SHORT_BINUNICODE: u8 = 0x8c;
    pub const BINUNICODE8: u8 = 0x8d;
    pub const EMPTY_TUPLE: u8 = b')';
    pub const TUPLE: u8 = b't';
    pub const TUPLE1: u8 = 0x85;
    pub const TUPLE2: u8 = 0x86;
    pub const TUPLE3: u8 = 0x87;
    pub const EMPTY_LIST: u8 = b']';
    pub const APPEND: u8 = b'a';
    pub const APPENDS: u8 = b'e';
    pub const EMPTY_DICT: u8 = b'}';
    pub const SETITEM: u8 = b's';
    pub const SETITEMS: u8 = b'u';
    pub const GLOBAL: u8 = b'c';
    pub const STACK_GLOBAL: u8 = 0x93;
    pub const REDUCE: u8 = b'R';
    pub const BUILD: u8 = b'b';
    pub const BINPERSID: u8 = b'Q';
}

/// The state of the parser.
struct Parser<'a> {
    data: &'a [u8],
    pos: usize,
    nodes: Vec<Node>,
    stack: Vec<Id>,
    /// Where each open `MARK` frame starts on the stack.
    marks: Vec<usize>,
    memo: HashMap<u32, Id>,
    strings: HashMap<Rc<str>, Id>,
}

impl<'a> Parser<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.data.len())
            .ok_or_else(|| bad("truncated"))?;
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().unwrap())
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    fn line(&mut self) -> Result<&'a str> {
        let rest = &self.data[self.pos..];
        let n = rest
            .iter()
            .position(|&b| b == b'\n')
            .ok_or_else(|| bad("truncated"))?;
        let line = self.take(n + 1)?;
        std::str::from_utf8(&line[..n]).map_err(|_| bad("a global's name is not UTF-8"))
    }

    fn add(&mut self, node: Node) -> Result<Id> {
        let id = Id::try_from(self.nodes.len()).map_err(|_| bad("too many objects"))?;
        self.nodes.push(node);
        Ok(id)
    }

    fn push(&mut self, node: Node) -> Result<()> {
        let id = self.add(node)?;
        self.stack.push(id);
        Ok(())
    }

    fn frame_start(&self) -> usize {
        self.marks.last().copied().unwrap_or(0)
    }

    fn pop(&mut self) -> Result<Id> {
        if self.stack.len() <= self.frame_start() {
            return Err(bad("stack underflow"));
        }
        Ok(self.stack.pop().unwrap())
    }

    fn top(&self) -> Result<Id> {
        if self.stack.len() <= self.frame_start() {
            return Err(bad("stack underflow"));
        }
        Ok(*self.stack.last().unwrap())
    }

    fn pop_mark(&mut self) -> Result<Vec<Id>> {
        let start = self.marks.pop().ok_or_else(|| bad("no MARK"))?;
        Ok(self.stack.split_off(start))
    }

    fn string(&mut self, len: u64) -> Result<()> {
        let len = usize::try_from(len).map_err(|_| bad("truncated"))?;
        let bytes = self.take(len)?;
        let s = std::str::from_utf8(bytes).map_err(|_| bad("a string is not UTF-8"))?;
        let id = match self.strings.get(s) {
            Some(&id) => id,
            None => {
                let s: Rc<str> = s.into();
                let id = self.add(Node::Str(s.clone()))?;
                self.strings.insert(s, id);
                id
            }
        };
        self.stack.push(id);
        Ok(())
    }

    fn memoize(&mut self, index: u32) -> Result<()> {
        let top = self.top()?;
        if self.memo.len() >= MEMO_LIMIT && !self.memo.contains_key(&index) {
            return Err(unsupported("too many memo entries"));
        }
        self.memo.insert(index, top);
        Ok(())
    }

    fn global(&mut self, module: &str, name: &str) -> Result<()> {
        let global = Global::find(module, name)?;
        self.push(Node::Global(global))
    }

    fn str_node(&self, id: Id) -> Result<&str> {
        match &self.nodes[id as usize] {
            Node::Str(s) => Ok(s),
            other => Err(bad(&format!(
                "expected a string for STACK_GLOBAL, found a {}",
                other.type_name()
            ))),
        }
    }

    fn key(&self, id: Id) -> Result<KeyId> {
        match &self.nodes[id as usize] {
            Node::Int(i) => Ok(KeyId::Int(*i)),
            Node::Str(_) => Ok(KeyId::Str(id)),
            other => Err(unsupported(&format!(
                "a dict key is a {}; only int and str keys are supported",
                other.type_name()
            ))),
        }
    }

    fn set_items(&mut self, target: Id, items: &[Id]) -> Result<()> {
        let keys = items
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[key, value]| Ok((self.key(key)?, value)))
            .collect::<Result<Vec<_>>>()?;
        let Node::Dict(dict) = &mut self.nodes[target as usize] else {
            return Err(bad(&format!(
                "SETITEM on a {}",
                self.nodes[target as usize].type_name()
            )));
        };
        for (key, value) in keys {
            match dict.index.get(&key) {
                Some(&i) => dict.items[i].1 = value,
                None => {
                    dict.index.insert(key, dict.items.len());
                    dict.items.push((key, value));
                }
            }
        }
        Ok(())
    }

    fn append(&mut self, target: Id, items: Vec<Id>) -> Result<()> {
        let node = &mut self.nodes[target as usize];
        let Node::List(list) = node else {
            return Err(bad(&format!("APPEND on a {}", node.type_name())));
        };
        list.extend(items);
        Ok(())
    }

    fn tuple(&mut self, items: Vec<Id>) -> Result<()> {
        self.push(Node::Tuple(items))
    }

    fn reduce(&mut self) -> Result<()> {
        let args = self.pop()?;
        let func = self.top()?;
        let Node::Tuple(arg_items) = &self.nodes[args as usize] else {
            return Err(bad("REDUCE arguments are not a tuple"));
        };
        let node = match self.nodes[func as usize] {
            Node::Global(Global::OrderedDict) => {
                if !arg_items.is_empty() {
                    return Err(unsupported("OrderedDict called with arguments"));
                }
                Node::Dict(DictNode {
                    ordered: true,
                    ..DictNode::default()
                })
            }
            Node::Global(func @ (Global::RebuildTensorV2 | Global::RebuildParameter)) => {
                Node::Call { func, args }
            }
            Node::Global(global) => {
                return Err(unsupported(&format!("{} can't be called", global.name())));
            }
            ref other => {
                return Err(bad(&format!("REDUCE calls a {}", other.type_name())));
            }
        };
        let id = self.add(node)?;
        *self.stack.last_mut().unwrap() = id;
        Ok(())
    }

    fn build(&mut self) -> Result<()> {
        let state = self.pop()?;
        let inst = self.top()?;
        let inst_node = &self.nodes[inst as usize];
        if !matches!(inst_node, Node::Dict(dict) if dict.ordered) {
            return Err(unsupported(&format!(
                "BUILD on a {}; only an OrderedDict's attributes are supported",
                inst_node.type_name()
            )));
        }
        // The attributes are bound now: changing the state dict later doesn't change them.
        let Node::Dict(attributes) = &self.nodes[state as usize] else {
            return Err(unsupported(&format!(
                "the attributes of an OrderedDict are a {}",
                self.nodes[state as usize].type_name()
            )));
        };
        let mut metadata = None;
        for &(key, value) in &attributes.items {
            match key {
                KeyId::Str(id) if self.str_node(id)? == "_metadata" => metadata = Some(value),
                KeyId::Str(id) => {
                    return Err(unsupported(&format!(
                        "the OrderedDict attribute {:?}; only _metadata is supported",
                        self.str_node(id)?
                    )));
                }
                KeyId::Int(i) => {
                    return Err(unsupported(&format!(
                        "the OrderedDict attribute {i}; only _metadata is supported"
                    )));
                }
            }
        }
        if let Some(metadata) = metadata {
            let Node::Dict(dict) = &mut self.nodes[inst as usize] else {
                unreachable!("checked above");
            };
            dict.metadata = Some(metadata);
        }
        Ok(())
    }

    fn run(mut self) -> Result<Pickle> {
        loop {
            let at = self.pos;
            let code = self.byte()?;
            match code {
                op::PROTO => {
                    let protocol = self.byte()?;
                    if !(2..=5).contains(&protocol) {
                        return Err(unsupported(&format!("protocol {protocol}")));
                    }
                }
                op::FRAME => {
                    let len = u64::from_le_bytes(self.array()?);
                    if len > (self.data.len() - self.pos) as u64 {
                        return Err(bad("truncated"));
                    }
                }
                op::STOP => {
                    let root = self.pop()?;
                    return Ok(Pickle {
                        nodes: self.nodes,
                        root,
                        input_len: self.data.len(),
                    });
                }
                op::MARK => self.marks.push(self.stack.len()),
                op::POP => {
                    if self.stack.len() > self.frame_start() {
                        self.stack.pop();
                    } else {
                        self.pop_mark()?;
                    }
                }
                op::POP_MARK => {
                    self.pop_mark()?;
                }
                op::MEMOIZE => {
                    let index = u32::try_from(self.memo.len()).unwrap();
                    self.memoize(index)?;
                }
                op::BINPUT => {
                    let index = self.byte()?.into();
                    self.memoize(index)?;
                }
                op::LONG_BINPUT => {
                    let index = u32::from_le_bytes(self.array()?);
                    self.memoize(index)?;
                }
                op::BINGET | op::LONG_BINGET => {
                    let index = if code == op::BINGET {
                        self.byte()?.into()
                    } else {
                        u32::from_le_bytes(self.array()?)
                    };
                    let id = *self
                        .memo
                        .get(&index)
                        .ok_or_else(|| bad(&format!("memo entry {index} is not defined")))?;
                    self.stack.push(id);
                }
                op::NONE => self.push(Node::None)?,
                op::NEWTRUE => self.push(Node::Bool(true))?,
                op::NEWFALSE => self.push(Node::Bool(false))?,
                op::BININT => {
                    let i = i32::from_le_bytes(self.array()?);
                    self.push(Node::Int(i.into()))?;
                }
                op::BININT1 => {
                    let i = self.byte()?;
                    self.push(Node::Int(i.into()))?;
                }
                op::BININT2 => {
                    let i = u16::from_le_bytes(self.array()?);
                    self.push(Node::Int(i.into()))?;
                }
                op::LONG1 => {
                    let n = self.byte()? as usize;
                    if n > 8 {
                        return Err(unsupported("an integer is outside the range of i64"));
                    }
                    let bytes = self.take(n)?;
                    let mut buf = if bytes.last().is_some_and(|&b| b & 0x80 != 0) {
                        [0xff; 8]
                    } else {
                        [0; 8]
                    };
                    buf[..n].copy_from_slice(bytes);
                    self.push(Node::Int(i64::from_le_bytes(buf)))?;
                }
                op::BINFLOAT => {
                    let f = f64::from_be_bytes(self.array()?);
                    self.push(Node::Float(f))?;
                }
                op::BINUNICODE => {
                    let len = u32::from_le_bytes(self.array()?);
                    self.string(len.into())?;
                }
                op::SHORT_BINUNICODE => {
                    let len = self.byte()?;
                    self.string(len.into())?;
                }
                op::BINUNICODE8 => {
                    let len = u64::from_le_bytes(self.array()?);
                    self.string(len)?;
                }
                op::EMPTY_TUPLE => self.tuple(vec![])?,
                op::TUPLE => {
                    let items = self.pop_mark()?;
                    self.tuple(items)?;
                }
                op::TUPLE1 | op::TUPLE2 | op::TUPLE3 => {
                    let n = (code - op::TUPLE1 + 1) as usize;
                    if self.stack.len() < self.frame_start() + n {
                        return Err(bad("stack underflow"));
                    }
                    let items = self.stack.split_off(self.stack.len() - n);
                    self.tuple(items)?;
                }
                op::EMPTY_LIST => self.push(Node::List(vec![]))?,
                op::APPEND => {
                    let item = self.pop()?;
                    let target = self.top()?;
                    self.append(target, vec![item])?;
                }
                op::APPENDS => {
                    let items = self.pop_mark()?;
                    let target = self.top()?;
                    self.append(target, items)?;
                }
                op::EMPTY_DICT => self.push(Node::Dict(DictNode::default()))?,
                op::SETITEM => {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    let target = self.top()?;
                    self.set_items(target, &[key, value])?;
                }
                op::SETITEMS => {
                    let items = self.pop_mark()?;
                    if items.len() % 2 != 0 {
                        return Err(bad("SETITEMS with an odd number of items"));
                    }
                    let target = self.top()?;
                    self.set_items(target, &items)?;
                }
                op::GLOBAL => {
                    let module = self.line()?;
                    let name = self.line()?;
                    self.global(module, name)?;
                }
                op::STACK_GLOBAL => {
                    let name = self.pop()?;
                    let module = self.pop()?;
                    let (module, name) = (self.str_node(module)?, self.str_node(name)?);
                    let global = Global::find(module, name)?;
                    self.push(Node::Global(global))?;
                }
                op::REDUCE => self.reduce()?,
                op::BUILD => self.build()?,
                op::BINPERSID => {
                    let pid = self.pop()?;
                    self.push(Node::Persistent(pid))?;
                }
                _ => {
                    return Err(unsupported(&format!(
                        "opcode 0x{code:02x} at byte {at} is not supported"
                    )));
                }
            }
        }
    }
}

impl Pickle {
    /// Parses a pickle, checking its structure but not yet what its objects mean.
    pub(super) fn parse(data: &[u8]) -> Result<Pickle> {
        Parser {
            data,
            pos: 0,
            nodes: vec![],
            stack: vec![],
            marks: vec![],
            memo: HashMap::new(),
            strings: HashMap::new(),
        }
        .run()
    }

    pub(super) fn node(&self, id: Id) -> &Node {
        &self.nodes[id as usize]
    }

    /// The budget of values for an input of this size (see [`to_value`](Pickle::to_value)), for
    /// charging other work whose amount shared references can multiply.
    pub(super) fn value_budget(&self) -> u64 {
        budgets(self.input_len).0
    }

    /// The object that the pickle returns, as a [`Value`], with tensors from `tensors`.
    pub(super) fn to_value(&self, tensors: &mut dyn Tensors) -> Result<Value> {
        let (nodes, bytes) = budgets(self.input_len);
        let mut m = Materializer {
            pickle: self,
            tensors,
            nodes_left: nodes,
            bytes_left: bytes,
            active: vec![false; self.nodes.len()],
        };
        m.charge_nodes(1)?;
        m.value(self.root, 0)
    }
}

struct Materializer<'a> {
    pickle: &'a Pickle,
    tensors: &'a mut dyn Tensors,
    nodes_left: u64,
    bytes_left: u64,
    /// The containers being materialized: meeting one again is a cycle.
    active: Vec<bool>,
}

/// The error for running out of a budget.
pub(super) fn over_budget() -> Error {
    unsupported(
        "the objects expand to much more than the size of the file, through shared references",
    )
}

impl Materializer<'_> {
    fn charge_nodes(&mut self, n: usize) -> Result<()> {
        self.nodes_left = self
            .nodes_left
            .checked_sub(n as u64)
            .ok_or_else(over_budget)?;
        Ok(())
    }

    fn string(&mut self, s: &str) -> Result<String> {
        self.bytes_left = self
            .bytes_left
            .checked_sub(s.len() as u64)
            .ok_or_else(over_budget)?;
        Ok(s.to_string())
    }

    fn key(&mut self, key: KeyId) -> Result<Key> {
        Ok(match key {
            KeyId::Int(i) => Key::Int(i),
            KeyId::Str(id) => {
                let Node::Str(s) = self.pickle.node(id) else {
                    unreachable!("string keys are string nodes");
                };
                Key::Str(self.string(s)?)
            }
        })
    }

    /// Materializes the children of a container, whose count is already charged.
    fn items(&mut self, items: &[Id], depth: usize) -> Result<Vec<Value>> {
        self.charge_nodes(items.len())?;
        let mut out = Vec::with_capacity(items.len());
        for &item in items {
            out.push(self.value(item, depth + 1)?);
        }
        Ok(out)
    }

    fn value(&mut self, id: Id, depth: usize) -> Result<Value> {
        let pickle = self.pickle;
        let node = pickle.node(id);
        let container = matches!(node, Node::Tuple(_) | Node::List(_) | Node::Dict(_));
        if container {
            if self.active[id as usize] {
                return Err(unsupported("recursive pickle structures are not supported"));
            }
            if depth >= MAX_DEPTH {
                return Err(unsupported(&format!(
                    "objects are nested more than {MAX_DEPTH} levels deep"
                )));
            }
            self.active[id as usize] = true;
        }
        let value = match node {
            Node::None => Value::None,
            Node::Bool(b) => Value::Bool(*b),
            Node::Int(i) => Value::Int(*i),
            Node::Float(f) => Value::Float(*f),
            Node::Str(s) => Value::Str(self.string(s)?),
            Node::Tuple(items) => Value::Tuple(self.items(items, depth)?),
            Node::List(items) => Value::List(self.items(items, depth)?),
            Node::Dict(dict) => Value::Dict(self.dict(dict, depth)?),
            Node::Call { func, args } => {
                Value::Tensor(self.tensors.tensor(pickle, id, *func, *args)?)
            }
            Node::Global(global) => {
                return Err(unsupported(&format!(
                    "{} is not a supported value",
                    global.name()
                )));
            }
            Node::Persistent(pid) => {
                return Err(unsupported(&format!(
                    "a persistent id ({}) outside a tensor",
                    pickle.node(*pid).type_name()
                )));
            }
        };
        if container {
            self.active[id as usize] = false;
        }
        Ok(value)
    }

    fn dict(&mut self, node: &DictNode, depth: usize) -> Result<Dict> {
        let mut dict = if node.ordered {
            Dict::new_ordered()
        } else {
            Dict::new()
        };
        self.charge_nodes(node.items.len())?;
        for &(key, value) in &node.items {
            let key = self.key(key)?;
            let value = self.value(value, depth + 1)?;
            dict.insert(key, value);
        }
        if let Some(metadata) = node.metadata {
            dict.set_versions(Some(self.versions(metadata)?));
        }
        Ok(dict)
    }

    /// The versions in an `OrderedDict`'s `_metadata` attribute: `{path: {"version": n}}`.
    /// Other fields of the entries are ignored; an entry without `version` gives no version.
    fn versions(&mut self, metadata: Id) -> Result<ModuleVersions> {
        let pickle = self.pickle;
        let Node::Dict(metadata) = pickle.node(metadata) else {
            return Err(unsupported("_metadata is not a dict"));
        };
        self.charge_nodes(metadata.items.len())?;
        let mut versions = ModuleVersions::new();
        for &(path, entry) in &metadata.items {
            let Key::Str(path) = self.key(path)? else {
                return Err(unsupported("a _metadata key is not a string"));
            };
            let Node::Dict(entry) = pickle.node(entry) else {
                return Err(unsupported(&format!("_metadata[{path:?}] is not a dict")));
            };
            let version = entry.items.iter().find(|(key, _)| {
                matches!(key, KeyId::Str(id) if matches!(pickle.node(*id), Node::Str(s) if &**s == "version"))
            });
            if let Some(&(_, version)) = version {
                let version = match pickle.node(version) {
                    Node::Int(v) => u32::try_from(*v).ok(),
                    _ => None,
                }
                .ok_or_else(|| {
                    unsupported(&format!(
                        "the version of {path:?} in _metadata is not an integer from 0 to {}",
                        u32::MAX
                    ))
                })?;
                versions.insert(path, version);
            }
        }
        Ok(versions)
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    /// Stands in for tensors: an empty tensor for each call, counting them.
    #[derive(Default)]
    pub(crate) struct Placeholders(pub(crate) usize);

    impl Tensors for Placeholders {
        fn tensor(&mut self, pickle: &Pickle, call: Id, func: Global, args: Id) -> Result<Tensor> {
            assert!(matches!(pickle.node(call), Node::Call { .. }));
            assert!(matches!(
                func,
                Global::RebuildTensorV2 | Global::RebuildParameter
            ));
            assert!(matches!(pickle.node(args), Node::Tuple(_)));
            self.0 += 1;
            Ok(crate::zeros([0]))
        }
    }

    /// The canonical form of `tools/golden/gen_m3_pt.py`.
    pub(crate) fn canonical(v: &Value) -> String {
        let hex = |s: &str| s.bytes().map(|b| format!("{b:02x}")).collect::<String>();
        let key = |k: &Key| match k {
            Key::Int(i) => format!("i{i}"),
            Key::Str(s) => format!("s{}", hex(s)),
        };
        let list = |items: &[Value]| items.iter().map(canonical).collect::<Vec<_>>().join(",");
        match v {
            Value::None => "N".into(),
            Value::Bool(true) => "T".into(),
            Value::Bool(false) => "F".into(),
            Value::Int(i) => format!("i{i}"),
            Value::Float(f) => format!("f{:016x}", f.to_bits()),
            Value::Str(s) => format!("s{}", hex(s)),
            Value::Tensor(_) => "t".into(),
            Value::List(items) => format!("[{}]", list(items)),
            Value::Tuple(items) => format!("({})", list(items)),
            Value::Dict(d) => {
                let items = d
                    .iter()
                    .map(|(k, v)| format!("{}:{}", key(k), canonical(v)))
                    .collect::<Vec<_>>()
                    .join(",");
                if !d.is_ordered() {
                    return format!("{{{items}}}");
                }
                let versions = d.versions().map_or(String::new(), |versions| {
                    let entries: Vec<String> = versions
                        .iter()
                        .map(|(path, v)| format!("s{}={v}", hex(path)))
                        .collect();
                    format!("<{}>", entries.join(";"))
                });
                format!("O{versions}{{{items}}}")
            }
        }
    }

    /// Parses protocol 2 opcodes and materializes the result.
    fn load(ops: &[u8]) -> Result<Value> {
        let mut data = vec![op::PROTO, 2];
        data.extend_from_slice(ops);
        Pickle::parse(&data)?.to_value(&mut Placeholders::default())
    }

    fn value(ops: &[u8]) -> String {
        canonical(&load(ops).unwrap())
    }

    fn err(ops: &[u8], message: &str) {
        match load(ops) {
            Err(Error::Format(m)) if m.contains(message) => {}
            other => panic!("expected a format error with {message:?}, got {other:?}"),
        }
    }

    fn str_op(s: &str) -> Vec<u8> {
        let mut v = vec![op::BINUNICODE];
        v.extend_from_slice(&(s.len() as u32).to_le_bytes());
        v.extend_from_slice(s.as_bytes());
        v
    }

    fn cat(parts: &[&[u8]]) -> Vec<u8> {
        parts.concat()
    }

    #[test]
    fn memo_references_are_the_same_object() {
        // L = []; t = (L, L); L.append(7); t  ->  ([7], [7])
        assert_eq!(
            value(b"]q\x00h\x00\x86q\x010h\x00K\x07a0h\x01."),
            "([i7],[i7])"
        );
        // A sparse memo index allocates nothing.
        assert_eq!(
            value(b"]r\xff\xff\xff\xffj\xff\xff\xff\xff\x86."),
            "([],[])"
        );
        // MEMOIZE numbers entries by the memo's size.
        assert_eq!(value(b"K\x05\x94K\x06\x94h\x00h\x01\x86."), "(i5,i6)");
        err(b"h\x05.", "memo entry 5 is not defined");
    }

    #[test]
    fn dicts_take_int_and_str_keys_and_keep_positions() {
        let a = str_op("a");
        let b = str_op("b");
        let zero = str_op("0");
        // {"a": 1, "b": 2, "a": 3}: the second "a" keeps the first one's position.
        let ops = cat(&[b"}(", &a, b"K\x01", &b, b"K\x02", &a, b"K\x03u."]);
        assert_eq!(value(&ops), "{s61:i3,s62:i2}");
        // 0 and "0" are different keys.
        let ops = cat(&[b"}(K\x00K\x01", &zero, b"K\x02u."]);
        assert_eq!(value(&ops), "{i0:i1,s30:i2}");
        err(b"}\x88K\x01s.", "a dict key is a bool");
        err(
            b"}G\x3f\xf0\x00\x00\x00\x00\x00\x00K\x01s.",
            "a dict key is a float",
        );
        err(b"}(K\x01u.", "odd number");
        err(b"]K\x01K\x02s.", "SETITEM on a list");
    }

    #[test]
    fn scalars() {
        assert_eq!(
            value(b"\x8a\x00\x8a\x01\xff\x8a\x02\x00\x80\x8a\x08\x00\x00\x00\x00\x00\x00\x00\x80\x87."),
            "(i-1,i-32768,i-9223372036854775808)"
        );
        assert_eq!(value(b"\x8a\x00."), "i0");
        assert_eq!(
            value(b"J\xff\xff\xff\xffK\xffM\xff\xff\x87."),
            "(i-1,i255,i65535)"
        );
        assert_eq!(
            value(b"G\xbf\xf8\x00\x00\x00\x00\x00\x00."),
            format!("f{:016x}", (-1.5f64).to_bits())
        );
        assert_eq!(value(b"N\x88\x89\x87."), "(N,T,F)");
        assert_eq!(value(b"\x8c\x02\xc3\xa9."), "sc3a9");
        assert_eq!(value(b"\x8d\x01\x00\x00\x00\x00\x00\x00\x00x."), "s78");
        err(
            b"\x8a\x09\x00\x00\x00\x00\x00\x00\x00\x00\x01.",
            "outside the range of i64",
        );
        err(b"\x8c\x01\xff.", "not UTF-8");
        // A surrogate, which Python's "surrogatepass" would accept.
        err(b"\x8c\x03\xed\xa0\x80.", "not UTF-8");
    }

    #[test]
    fn marks_frame_the_stack() {
        assert_eq!(value(b"(K\x01(1K\x02t."), "(i1,i2)");
        // POP on an empty frame pops the mark, as in Python.
        assert_eq!(value(b"K\x01(0."), "i1");
        // Inside a frame, the stack below the mark can't be reached.
        err(b"](K\x01a.", "stack underflow");
        err(b"1.", "no MARK");
        err(b".", "stack underflow");
        // STOP takes the top of the current frame, which is empty here.
        err(b"K\x01(.", "stack underflow");
        err(b"K\x01\x86.", "stack underflow");
        err(b"(K\x01\x86.", "stack underflow");
    }

    #[test]
    fn truncated_and_unsupported_input() {
        let good = b"}(X\x01\x00\x00\x00aK\x01u.";
        for n in 0..good.len() {
            assert!(load(&good[..n]).is_err(), "{n}");
        }
        err(b"S'a'\n.", "opcode 0x53 at byte 2");
        err(b"\x95\xff\x00\x00\x00\x00\x00\x00\x00N.", "truncated");
        for protocol in [0u8, 1, 6] {
            let data = [op::PROTO, protocol, b'N', b'.'];
            assert!(Pickle::parse(&data).is_err(), "{protocol}");
        }
        // Protocols 4 and 5 add FRAME, MEMOIZE, SHORT_BINUNICODE and STACK_GLOBAL.
        let data = b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x01a\x94h\x00\x86.";
        let value = Pickle::parse(data)
            .unwrap()
            .to_value(&mut Placeholders::default());
        assert_eq!(canonical(&value.unwrap()), "(s61,s61)");
    }

    #[test]
    fn only_listed_globals_and_calls() {
        err(b"cos\nsystem\n.", "the global os.system is not supported");
        err(b"ctorch\nSize\n.", "the global torch.Size is not supported");
        err(b"cbuiltins\nset\n.", "builtins.set");
        err(b"K\x01K\x02\x93.", "expected a string for STACK_GLOBAL");
        let ops = cat(&[&str_op("os"), &str_op("system"), b"\x93."]);
        err(&ops, "os.system");
        err(
            b"ctorch\nFloatStorage\n)R.",
            "torch.FloatStorage can't be called",
        );
        err(
            b"ccollections\nOrderedDict\nK\x01\x85R.",
            "OrderedDict called with arguments",
        );
        err(
            b"ccollections\nOrderedDict\nK\x01R.",
            "REDUCE arguments are not a tuple",
        );
        err(b"K\x01)R.", "REDUCE calls a int");
        // A global or a persistent id is not a value; a call of a rebuild function is a tensor.
        err(
            b"ctorch\nFloatStorage\n.",
            "torch.FloatStorage is not a supported value",
        );
        err(b"K\x01Q.", "a persistent id (int) outside a tensor");
        assert_eq!(value(b"ctorch._utils\n_rebuild_tensor_v2\n)R."), "t");
        assert_eq!(value(b"ctorch._utils\n_rebuild_parameter\n)R."), "t");
        err(b"ctorch._utils\n_rebuild_tensor_v2\nK\x01R.", "not a tuple");
    }

    /// `OrderedDict()` with the attributes `state`.
    fn ordered_with(state: &[u8]) -> Vec<u8> {
        cat(&[b"ccollections\nOrderedDict\n)R", state, b"b."])
    }

    #[test]
    fn ordered_dict_metadata() {
        let entry = |path: &str, version: &[u8]| {
            cat(&[&str_op(path), b"}", &str_op("version"), version, b"s"])
        };
        let metadata = |entries: &[u8]| cat(&[b"}", &str_op("_metadata"), b"}(", entries, b"us"]);
        // The version of each path; entries without one and other fields are skipped.
        let entries = cat(&[
            &entry("", b"K\x01"),
            &entry("bn", b"K\x02"),
            &str_op("none"),
            b"}",
            &str_op("other"),
            b"K\x05s",
        ]);
        let v = load(&ordered_with(&metadata(&entries))).unwrap();
        let Value::Dict(d) = &v else { panic!() };
        let versions: Vec<_> = d.versions().unwrap().iter().collect();
        assert_eq!(versions, [("", 1), ("bn", 2)]);
        // Without _metadata, an empty state gives no versions.
        let v = load(&ordered_with(b"}")).unwrap();
        let Value::Dict(d) = &v else { panic!() };
        assert!(d.versions().is_none() && d.is_ordered());
        for (version, message) in [
            (
                &b"J\xff\xff\xff\xff"[..],
                "not an integer from 0 to 4294967295",
            ),
            (b"\x8a\x05\x00\x00\x00\x00\x01", "not an integer"),
            (b"N", "not an integer"),
        ] {
            err(&ordered_with(&metadata(&entry("x", version))), message);
        }
        err(
            &ordered_with(&cat(&[b"}", &str_op("other"), b"K\x01s"])),
            "the OrderedDict attribute \"other\"",
        );
        err(
            &ordered_with(b"K\x01"),
            "the attributes of an OrderedDict are a int",
        );
        err(
            &ordered_with(&cat(&[b"}", &str_op("_metadata"), b"K\x01s"])),
            "_metadata is not a dict",
        );
        err(
            &ordered_with(&metadata(&cat(&[&str_op("x"), b"K\x01"]))),
            "_metadata[\"x\"] is not a dict",
        );
        err(b"}}b.", "BUILD on a dict");
    }

    /// Each `BUILD` binds the attributes its state has when it runs, like
    /// `inst.__dict__.update(state)` in the reference (R114; the files are
    /// tests/golden/m3_pt/build_*.pt).
    #[test]
    fn build_binds_attributes_when_it_runs() {
        let metadata =
            |version: &[u8]| cat(&[b"}", &str_op(""), b"}", &str_op("version"), version, b"ss"]);
        let version = |ops: &[u8]| {
            let v = load(ops).unwrap();
            let Value::Dict(d) = &v else { panic!("{v:?}") };
            d.versions()
                .map(|v| v.iter().map(|(p, v)| (p.to_owned(), v)).collect::<Vec<_>>())
        };
        // OrderedDict() in memo 2, BUILD with {"_metadata": {"": {"version": 1}}} in memo 0.
        let od = b"ccollections\nOrderedDict\n)Rq\x02";
        let first = cat(&[b"}q\x00", &str_op("_metadata"), &metadata(b"K\x01"), b"sb"]);
        let get = |memo: u8| [op::BINGET, memo];
        for (ops, expected) in [
            (cat(&[od, &first, b"."]), 1),
            // Rebinding _metadata in the state dict after BUILD changes nothing...
            (
                cat(&[
                    od,
                    &first,
                    &get(0),
                    &str_op("_metadata"),
                    &metadata(b"K\x02"),
                    b"s",
                    &get(2),
                    b".",
                ]),
                1,
            ),
            // ...nor does adding an attribute that BUILD would reject.
            (
                cat(&[
                    od,
                    &first,
                    &get(0),
                    &str_op("other"),
                    b"K\x01s",
                    &get(2),
                    b".",
                ]),
                1,
            ),
            // A state without _metadata keeps the bound one; another _metadata replaces it.
            (cat(&[od, &first, b"}b."]), 1),
            (
                cat(&[
                    od,
                    &first,
                    b"}",
                    &str_op("_metadata"),
                    &metadata(b"K\x02"),
                    b"sb.",
                ]),
                2,
            ),
            // The bound metadata dict is shared: changing it after BUILD is seen.
            (
                cat(&[
                    od,
                    b"}",
                    &str_op("_metadata"),
                    &metadata(b"K\x01"),
                    b"q\x01sb",
                    &get(1),
                    &str_op(""),
                    b"}",
                    &str_op("version"),
                    b"K\x02ss",
                    &get(2),
                    b".",
                ]),
                2,
            ),
        ] {
            assert_eq!(version(&ops), Some(vec![(String::new(), expected)]));
        }
        // The state is checked when BUILD runs, even if the OrderedDict is dropped later.
        err(
            &cat(&[od, b"}K\x01K\x01sb0N."]),
            "the OrderedDict attribute 1; only _metadata is supported",
        );
        err(
            &cat(&[od, b"]b0N."]),
            "the attributes of an OrderedDict are a list",
        );
    }

    #[test]
    fn cycles_depth_and_budgets() {
        // L = []; L.append(L)
        err(
            b"]q\x00h\x00a.",
            "recursive pickle structures are not supported",
        );
        err(b"}q\x00K\x01h\x00s.", "recursive");
        // A shared object that isn't a cycle is fine: (t, t) for t = (1,).
        assert_eq!(value(b"K\x01\x85q\x00h\x00\x86."), "((i1),(i1))");
        let nested = |n: usize| cat(&[b")", &vec![op::TUPLE1; n], b"."]);
        assert!(load(&nested(MAX_DEPTH - 1)).is_ok());
        err(&nested(MAX_DEPTH), "nested more than 256 levels deep");
        // Strings referenced many times count each time.
        let long = "x".repeat(1 << 16);
        let mut ops = cat(&[&str_op(&long), b"q\x00]("]);
        for _ in 0..40 {
            ops.extend_from_slice(b"h\x00");
        }
        ops.extend_from_slice(b"e.");
        err(&ops, "expand to much more than the size of the file");
        // So do containers: [[x, x], [x, x]] ... 30 levels deep.
        let mut ops = b"K\x00\x85q\x00".to_vec();
        for i in 0..30u8 {
            ops.extend_from_slice(&[b'h', i, b'h', i, op::TUPLE2, b'q', i + 1]);
        }
        ops.push(b'.');
        err(&ops, "expand to much more");
        // And dicts: {0: d, 1: d} ... 30 levels deep.
        let mut ops = b"}q\x00".to_vec();
        for i in 0..30u8 {
            ops.extend_from_slice(&[
                b'}',
                b'(',
                b'K',
                0,
                b'h',
                i,
                b'K',
                1,
                b'h',
                i,
                b'u',
                b'q',
                i + 1,
            ]);
        }
        ops.push(b'.');
        err(&ops, "expand to much more");
    }
}
