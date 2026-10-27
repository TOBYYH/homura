//! [`OptimizerStateDict`]: the options and state of an optimizer, as PyTorch's
//! `Optimizer.state_dict()` has them, and how the optimizers save and load it.

use std::collections::BTreeMap;
use std::fmt;

use super::{GroupOptions, Groups};
use crate::io::{Dict, Key, Value};
use crate::{DType, Device, Error, Result, Tensor};

/// The options of an optimizer's parameter groups and the state of its parameters, as PyTorch's
/// `Optimizer.state_dict()` returns them: what [`Optimizer::state_dict`](super::Optimizer::state_dict)
/// returns and [`Optimizer::load_state_dict`](super::Optimizer::load_state_dict) loads.
///
/// Parameters are numbered across the groups, in the order of the groups and of the parameters
/// in each group, from 0, as PyTorch numbers them. Values are [`Value`]s with PyTorch's names,
/// so that a state dict converts to and from the objects of PyTorch's `.pt` files:
/// `Value::from(&state_dict)` and `OptimizerStateDict::try_from(&value)`.
///
/// ```
/// use homura::io::{self, Value};
/// use homura::nn::{Linear, Module};
/// use homura::optim::{Adam, Optimizer, OptimizerStateDict};
///
/// let model = Linear::new(4, 2);
/// let mut opt = Adam::new(model.parameters(), 0.01);
/// model.forward(&homura::ones([8, 4])).pow(2.0).mean().backward();
/// opt.step();
///
/// // torch.save({"model": model.state_dict(), "optimizer": opt.state_dict()}, path)
/// let checkpoint: io::Dict = [
///     ("model", Value::from(&model.state_dict())),
///     ("optimizer", Value::from(&opt.state_dict())),
/// ]
/// .into_iter()
/// .collect();
/// let bytes = io::save_value_to_bytes(&Value::from(checkpoint))?;
///
/// let checkpoint = io::load_value_from_bytes(&bytes)?;
/// let Value::Dict(checkpoint) = checkpoint else { unreachable!() };
/// let state_dict = OptimizerStateDict::try_from(checkpoint.get("optimizer").unwrap())?;
/// let mut resumed = Adam::new(model.parameters(), 0.01);
/// resumed.load_state_dict(&state_dict)?;
/// assert_eq!(resumed.state_dict().state.len(), 2);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Debug, Default)]
pub struct OptimizerStateDict {
    /// The state of each parameter that has state, by its number (`state`): the values by name,
    /// in PyTorch's order, such as `momentum_buffer` for SGD.
    pub state: BTreeMap<usize, Vec<(String, Value)>>,
    /// The parameter groups, in order (`param_groups`).
    pub param_groups: Vec<ParamGroupRecord>,
}

/// A parameter group in an [`OptimizerStateDict`]: its options and the numbers of its
/// parameters.
#[derive(Clone, Debug, Default)]
pub struct ParamGroupRecord {
    /// The options by name, in PyTorch's order, such as `("lr", Value::Float(0.1))`.
    pub options: Vec<(String, Value)>,
    /// The numbers of the parameters (`params`).
    pub params: Vec<usize>,
}

/// Why [`Optimizer::load_state_dict`](super::Optimizer::load_state_dict) failed. Nothing was
/// changed.
///
/// `Display` lists the problems, one per line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadError {
    /// One message per problem, naming the parameter group and option, or the parameter number
    /// and state entry. When the groups don't match the optimizer's, only those problems.
    pub errors: Vec<String>,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Error(s) in loading optimizer state_dict:")?;
        for e in &self.errors {
            write!(f, "\n\t{e}")?;
        }
        Ok(())
    }
}

impl std::error::Error for LoadError {}

/// What an optimizer saves of the options `O` of a group and the state `S` of a parameter, and
/// how it reads them back.
pub(super) trait Schema<O, S> {
    /// The options of a group, with PyTorch's names and order, without `params`.
    fn save_options(&self, options: &O) -> Vec<(String, Value)>;

    /// The options of a group from its record, which [`Fields`] reads. They aren't validated
    /// yet.
    fn load_options(&self, fields: &mut Fields) -> O;

    /// The state of a parameter, with PyTorch's names and order.
    fn save_state(&self, state: &S) -> Vec<(String, Value)>;

    /// The state of the parameter `param` of a group with the options `options` from its record,
    /// which [`Fields`] reads: `None` if the record is empty or says there is none.
    fn load_state(&self, options: &O, param: &Tensor, fields: &mut Fields) -> Option<S>;
}

impl<O: GroupOptions, S> Groups<O, S> {
    pub(super) fn state_dict(&self, schema: &impl Schema<O, S>) -> OptimizerStateDict {
        let mut out = OptimizerStateDict::default();
        let mut next = 0;
        for (group, state) in self.groups.iter().zip(&self.state) {
            for (i, s) in state.iter().enumerate() {
                if let Some(s) = s {
                    out.state.insert(next + i, schema.save_state(s));
                }
            }
            out.param_groups.push(ParamGroupRecord {
                options: schema.save_options(&group.options),
                params: (next..next + group.params.len()).collect(),
            });
            next += group.params.len();
        }
        out
    }

    /// Checks everything first and changes nothing if something is wrong.
    pub(super) fn load_state_dict(
        &mut self,
        schema: &impl Schema<O, S>,
        state_dict: &OptimizerStateDict,
    ) -> std::result::Result<(), LoadError> {
        let records = &state_dict.param_groups;
        if records.len() != self.groups.len() {
            return Err(LoadError {
                errors: vec![format!(
                    "loaded state dict has a different number of parameter groups ({}, the \
                     optimizer has {})",
                    records.len(),
                    self.groups.len()
                )],
            });
        }
        let mut errors = vec![];
        for (g, (record, group)) in records.iter().zip(&self.groups).enumerate() {
            if record.params.len() != group.params.len() {
                errors.push(format!(
                    "loaded state dict contains a parameter group that doesn't match the size \
                     of optimizer's group: group {g} has {} parameters, the optimizer's {}",
                    record.params.len(),
                    group.params.len()
                ));
            }
        }
        fail_if_any(errors)?;

        // The group and position of each parameter number.
        let mut errors = vec![];
        let count = self.groups.iter().map(|g| g.params.len()).sum();
        let mut slots = vec![None; count];
        for (g, record) in records.iter().enumerate() {
            for (i, &n) in record.params.iter().enumerate() {
                match slots.get_mut(n) {
                    None => errors.push(format!(
                        "parameter group {g} lists the parameter number {n}, but there are \
                         {count} parameters"
                    )),
                    Some(Some(_)) => errors.push(format!(
                        "parameter group {g} lists the parameter number {n}, which was listed \
                         before"
                    )),
                    Some(slot) => *slot = Some((g, i)),
                }
            }
        }
        for &n in state_dict.state.keys().filter(|&&n| n >= count) {
            errors.push(format!(
                "the state has an entry for the parameter number {n}, but there are {count} \
                 parameters"
            ));
        }
        fail_if_any(errors)?;

        // The state depends on the options, so they are read first.
        let mut errors = vec![];
        let mut options = vec![];
        for (g, record) in records.iter().enumerate() {
            let at = format!("parameter group {g}");
            let mut fields = Fields::new(&record.options, &at, &mut errors);
            fields.skip_param_names(record.params.len());
            let o = schema.load_options(&mut fields);
            if fields.finish()
                && let Err(message) = o.validate()
            {
                errors.push(format!("{at}: {message}"));
            }
            options.push(o);
        }
        fail_if_any(errors)?;

        let mut errors = vec![];
        let mut state: Vec<Vec<Option<S>>> = self
            .groups
            .iter()
            .map(|g| g.params.iter().map(|_| None).collect())
            .collect();
        let _no_grad = crate::no_grad();
        for (&n, record) in &state_dict.state {
            // Every number has a slot: the groups list `count` distinct numbers below `count`.
            let (g, i) = slots[n].expect("every parameter number is listed");
            let at = format!("the state of parameter {n}");
            let mut fields = Fields::new(record, &at, &mut errors);
            let s = schema.load_state(&options[g], &self.groups[g].params[i], &mut fields);
            if fields.finish() {
                state[g][i] = s;
            }
        }
        fail_if_any(errors)?;

        for (group, o) in self.groups.iter_mut().zip(options) {
            group.options = o;
        }
        self.state = state;
        Ok(())
    }
}

fn fail_if_any(errors: Vec<String>) -> std::result::Result<(), LoadError> {
    if errors.is_empty() {
        Ok(())
    } else {
        Err(LoadError { errors })
    }
}

/// Reads the named values of a group's options or a parameter's state, collecting the problems:
/// names that appear twice, values of the wrong kind, and at the end names that weren't read.
/// The readers return a placeholder for a value with a problem.
pub(super) struct Fields<'a> {
    items: &'a [(String, Value)],
    read: Vec<bool>,
    at: &'a str,
    errors: &'a mut Vec<String>,
    failed: bool,
}

impl<'a> Fields<'a> {
    fn new(items: &'a [(String, Value)], at: &'a str, errors: &'a mut Vec<String>) -> Self {
        let mut fields = Fields {
            items,
            read: vec![false; items.len()],
            at,
            errors,
            failed: false,
        };
        for (i, (name, _)) in items.iter().enumerate() {
            if items[..i].iter().any(|(n, _)| n == name) {
                fields.error(name, "appears twice");
            }
        }
        fields
    }

    /// Records a problem with the value `name`.
    pub(super) fn error(&mut self, name: &str, message: impl fmt::Display) {
        self.errors.push(format!("{}: {name} {message}", self.at));
        self.failed = true;
    }

    /// Whether there are no values.
    pub(super) fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The value `name`, if there is one, marked as read.
    pub(super) fn get(&mut self, name: &str) -> Option<&'a Value> {
        let i = self.items.iter().position(|(n, _)| n == name)?;
        self.read[i] = true;
        Some(&self.items[i].1)
    }

    /// The value `name`; a problem if there is none.
    pub(super) fn required(&mut self, name: &str) -> Option<&'a Value> {
        let value = self.get(name);
        if value.is_none() {
            self.error(name, "is missing");
        }
        value
    }

    /// The number `name` (a Python `int` or `float`), or `default` if there is none.
    pub(super) fn number(&mut self, name: &str, default: Option<f64>) -> f64 {
        let value = match default {
            Some(_) => self.get(name),
            None => self.required(name),
        };
        match value {
            None => default.unwrap_or(0.0),
            Some(v) => self.as_number(name, v).unwrap_or(0.0),
        }
    }

    fn as_number(&mut self, name: &str, value: &Value) -> Option<f64> {
        match value {
            Value::Float(f) => Some(*f),
            // As Python's `float(i)`, rounded to nearest.
            Value::Int(i) => Some(*i as f64),
            v => {
                self.error(name, format_args!("is a {}, not a number", v.type_name()));
                None
            }
        }
    }

    /// The pair of numbers `name` (a `tuple` or `list`).
    pub(super) fn pair(&mut self, name: &str) -> (f64, f64) {
        match self.required(name) {
            Some(Value::Tuple(items) | Value::List(items)) if items.len() == 2 => {
                let first = self.as_number(name, &items[0]).unwrap_or(0.0);
                let second = self.as_number(name, &items[1]).unwrap_or(0.0);
                (first, second)
            }
            Some(v) => {
                self.error(
                    name,
                    format_args!("is a {}, not a pair of numbers", v.type_name()),
                );
                (0.0, 0.0)
            }
            None => (0.0, 0.0),
        }
    }

    /// The flag `name` (a Python `bool`), or `default` if there is none.
    pub(super) fn flag(&mut self, name: &str, default: bool) -> bool {
        match self.get(name) {
            None => default,
            Some(Value::Bool(b)) => *b,
            Some(v) => {
                self.error(name, format_args!("is a {}, not a bool", v.type_name()));
                default
            }
        }
    }

    /// An option that homura doesn't have: if there is one, it must be one of the values that
    /// select what homura does, described by `neutral`.
    pub(super) fn unsupported(&mut self, name: &str, accept: fn(&Value) -> bool, neutral: &str) {
        if let Some(v) = self.get(name)
            && !accept(v)
        {
            self.error(
                name,
                format_args!("is {v:?}; homura supports only {neutral}"),
            );
        }
    }

    /// The names of the parameters, which PyTorch saves when the optimizer was made with named
    /// parameters: read and not used.
    fn skip_param_names(&mut self, count: usize) {
        match self.get("param_names") {
            None => {}
            Some(Value::List(names))
                if names.len() == count && names.iter().all(|n| matches!(n, Value::Str(_))) => {}
            Some(_) => self.error(
                "param_names",
                format_args!("is not a list of {count} names, one per parameter"),
            ),
        }
    }

    /// Whether everything was read without a problem; records the values that weren't read.
    fn finish(mut self) -> bool {
        let unread: Vec<String> = self
            .items
            .iter()
            .zip(&self.read)
            .filter(|(_, read)| !**read)
            .map(|((name, _), _)| name.clone())
            .collect();
        for name in unread {
            self.error(&name, "is not known");
        }
        !self.failed
    }
}

/// A copy of the state tensor `name` for the parameter `param`: a floating point tensor of its
/// shape, converted to its dtype and device. The copy shares no storage with `value` and doesn't
/// require grad.
pub(super) fn load_moment(param: &Tensor, fields: &mut Fields, name: &str) -> Option<Tensor> {
    let value = fields.required(name)?;
    let Value::Tensor(t) = value else {
        fields.error(
            name,
            format_args!("is a {}, not a tensor", value.type_name()),
        );
        return None;
    };
    if !t.is_floating_point() {
        fields.error(
            name,
            format_args!("has dtype {}; it must be floating point", t.dtype()),
        );
        return None;
    }
    if t.shape() != param.shape() {
        fields.error(
            name,
            format_args!(
                "has shape {:?}, but the parameter has shape {:?}",
                t.shape(),
                param.shape()
            ),
        );
        return None;
    }
    if t.device() == Device::Meta && param.device() != Device::Meta {
        fields.error(
            name,
            format_args!(
                "is a meta tensor, which has no data to copy to a parameter on {}",
                param.device()
            ),
        );
        return None;
    }
    let t = t.detach();
    let converted = t.to((param.dtype(), param.device()));
    Some(if converted.is_same(&t) {
        t.deep_clone()
    } else {
        converted
    })
}

/// The largest step count that F64 holds exactly, `2^53`.
const MAX_STEP: f64 = 9_007_199_254_740_992.0;

/// A copy of Adam's step count: a 0-D CPU tensor of dtype F32 or F64, kept as it is, or a Python
/// number, which PyTorch before 1.12 saved and which becomes a tensor of the dtype that new state
/// would have. The count must be a whole number from 0 to `2^53`.
pub(super) fn load_step(fields: &mut Fields, scalar_dtype: DType) -> Option<Tensor> {
    let name = "step";
    let value = fields.required(name)?;
    let (count, tensor) = match value {
        Value::Tensor(t) => {
            if t.num_dims() != 0
                || t.device() != Device::Cpu
                || !matches!(t.dtype(), DType::F32 | DType::F64)
            {
                fields.error(
                    name,
                    format_args!(
                        "has shape {:?}, dtype {} and device {}; it must be a 0-D tensor of \
                         dtype F32 or F64 on the CPU",
                        t.shape(),
                        t.dtype(),
                        t.device()
                    ),
                );
                return None;
            }
            (t.to(DType::F64).item::<f64>(), Some(t))
        }
        Value::Int(i) => {
            // Check before converting: 2^53 + 1 rounds down to 2^53 as an f64.
            if !(0..=MAX_STEP as i64).contains(i) {
                fields.error(
                    name,
                    format_args!("is {i}; it must be a whole number from 0 to 2^53"),
                );
                return None;
            }
            (*i as f64, None)
        }
        Value::Float(f) => (*f, None),
        v => {
            fields.error(name, format_args!("is a {}, not a tensor", v.type_name()));
            return None;
        }
    };
    if !(0.0..=MAX_STEP).contains(&count) || count.fract() != 0.0 {
        fields.error(
            name,
            format_args!("is {count}; it must be a whole number from 0 to 2^53"),
        );
        return None;
    }
    Some(match tensor {
        Some(t) => t.detach().deep_clone(),
        None => {
            let t = crate::zeros_with([], (scalar_dtype, Device::Cpu));
            t.fill_(count);
            t
        }
    })
}

/// A `dict` with `state`, a `dict` of `dict`s by parameter number, and `param_groups`, a `list`
/// of `dict`s with the options and then `params`, as PyTorch's `state_dict()`. The tensors are
/// shared, not copied.
impl From<&OptimizerStateDict> for Value {
    fn from(state_dict: &OptimizerStateDict) -> Value {
        let state: Dict = state_dict
            .state
            .iter()
            .map(|(&n, values)| (Key::Int(n as i64), Value::Dict(dict(values))))
            .collect();
        let groups = state_dict.param_groups.iter().map(|group| {
            let mut d = dict(&group.options);
            let params = group.params.iter().map(|&n| Value::Int(n as i64));
            d.insert("params", Value::List(params.collect()));
            Value::Dict(d)
        });
        let out: Dict = [
            ("state", Value::Dict(state)),
            ("param_groups", Value::List(groups.collect())),
        ]
        .into_iter()
        .collect();
        Value::Dict(out)
    }
}

fn dict(values: &[(String, Value)]) -> Dict {
    values
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect()
}

/// An optimizer state dict from what PyTorch's `state_dict()` returns: a `dict` with exactly
/// `state` and `param_groups`. Only the structure is checked here: parameter numbers are
/// non-negative `int`s, names are `str`s and each group has a `params` list. Whether the options
/// and state suit an optimizer is checked by
/// [`load_state_dict`](super::Optimizer::load_state_dict). The tensors are shared, not copied.
///
/// # Errors
///
/// [`Error::Format`] naming where the value doesn't have that structure.
impl TryFrom<&Value> for OptimizerStateDict {
    type Error = Error;

    fn try_from(value: &Value) -> Result<OptimizerStateDict> {
        let top = as_dict(value, "")?;
        for (key, _) in top.iter() {
            if !matches!(key, Key::Str(k) if k == "state" || k == "param_groups") {
                return Err(not_a_state_dict(format_args!(
                    "it has the key {key:?}; it can have only \"state\" and \"param_groups\""
                )));
            }
        }
        let get = |name: &str| {
            top.get(name)
                .ok_or_else(|| not_a_state_dict(format_args!("it has no {name:?}")))
        };
        let mut out = OptimizerStateDict::default();
        for (key, values) in as_dict(get("state")?, "[\"state\"]")?.iter() {
            let at = format!("[\"state\"][{key:?}]");
            let n = number(key, &at)?;
            out.state.insert(n, named(as_dict(values, &at)?, &at)?);
        }
        let Value::List(groups) = get("param_groups")? else {
            return Err(not_a_state_dict(format_args!(
                "[\"param_groups\"] is a {}, not a list",
                get("param_groups")?.type_name()
            )));
        };
        for (g, group) in groups.iter().enumerate() {
            let at = format!("[\"param_groups\"][{g}]");
            let mut options = named(as_dict(group, &at)?, &at)?;
            let Some(i) = options.iter().position(|(k, _)| k == "params") else {
                return Err(not_a_state_dict(format_args!("{at} has no \"params\"")));
            };
            let (_, params) = options.remove(i);
            let at = format!("{at}[\"params\"]");
            let Value::List(params) = params else {
                return Err(not_a_state_dict(format_args!(
                    "{at} is a {}, not a list",
                    params.type_name()
                )));
            };
            let params = params
                .iter()
                .enumerate()
                .map(|(i, p)| match p {
                    Value::Int(n) => number(&Key::Int(*n), &format!("{at}[{i}]")),
                    p => Err(not_a_state_dict(format_args!(
                        "{at}[{i}] is a {}, not an int",
                        p.type_name()
                    ))),
                })
                .collect::<Result<_>>()?;
            out.param_groups.push(ParamGroupRecord { options, params });
        }
        Ok(out)
    }
}

fn not_a_state_dict(message: fmt::Arguments) -> Error {
    Error::Format(format!("expected an optimizer state dict: {message}"))
}

fn as_dict<'a>(value: &'a Value, at: &str) -> Result<&'a Dict> {
    match value {
        Value::Dict(d) => Ok(d),
        v => Err(not_a_state_dict(format_args!(
            "{} is a {}, not a dict",
            if at.is_empty() { "it" } else { at },
            v.type_name()
        ))),
    }
}

/// A parameter number.
fn number(key: &Key, at: &str) -> Result<usize> {
    match key {
        Key::Int(n) => usize::try_from(*n).map_err(|_| {
            not_a_state_dict(format_args!("{at}: the parameter number {n} is negative"))
        }),
        Key::Str(_) => Err(not_a_state_dict(format_args!(
            "{at}: the key is a str, not a parameter number"
        ))),
    }
}

/// The values of a dict with `str` keys, in order.
fn named(dict: &Dict, at: &str) -> Result<Vec<(String, Value)>> {
    dict.iter()
        .map(|(key, value)| match key {
            Key::Str(k) => Ok((k.clone(), value.clone())),
            Key::Int(i) => Err(not_a_state_dict(format_args!(
                "{at} has the key {i}, which is not a str"
            ))),
        })
        .collect()
}
