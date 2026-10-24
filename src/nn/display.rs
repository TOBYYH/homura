//! Printing module trees as PyTorch does ([`ModuleDisplay`]), and the Python spellings of values
//! that the layers' [`extra_repr`](super::Module::extra_repr) texts use.

use std::fmt;

use super::Module;
use super::module::{Item, walk};

/// A module tree printed as PyTorch prints a module (`print(model)`), from
/// [`Module::display`]:
///
/// ```text
/// Mlp(
///   (hidden): Linear(in_features=4, out_features=8, bias=True)
///   (output): Linear(in_features=8, out_features=2, bias=True)
/// )
/// ```
///
/// Each module is written as its [`type_name`](Module::type_name), then in parentheses its
/// [`extra_repr`](Module::extra_repr) and its submodules, one per line as `(name): ...`, indented
/// by two spaces; an `extra_repr` of one line without submodules stays on the module's line. The
/// submodules are listed by path: a submodule shared through an `Arc` appears under each of its
/// names, as in PyTorch. Like PyTorch's `nn.ModuleList`, a module whose type name is
/// `ModuleList` (a `Vec` or an array) writes consecutive elements of the same text once:
/// `(0-2): 3 x Linear(...)`.
///
/// The text is written when the value is formatted, with `Display` or `Debug` alike, calling
/// `extra_repr` of each module then.
pub struct ModuleDisplay<'a> {
    root: &'a dyn Module,
    /// The submodules under `root`, each before its own submodules.
    descendants: Vec<Node<'a>>,
}

/// A submodule in a [`ModuleDisplay`].
struct Node<'a> {
    /// The number of names in its key: 1 for a child of the root.
    depth: usize,
    name: String,
    module: &'a dyn Module,
}

impl<'a> ModuleDisplay<'a> {
    /// The tree under `module`, which [`Module::display`] gives for sized modules; this one also
    /// takes the `&dyn Module`s that [`named_modules`](Module::named_modules) lists.
    ///
    /// # Panics
    ///
    /// Like the traversals of [`Module`]: if names are invalid or the tree has a cycle through
    /// shared submodules. The tree is checked here, not when it is formatted.
    #[track_caller]
    pub fn new(module: &'a dyn Module) -> ModuleDisplay<'a> {
        let mut descendants = vec![];
        walk(module, false, &mut |item| {
            if let Item::Module(key, module) = item
                && !key.is_empty()
            {
                let name = key.rsplit('.').next().unwrap_or_default().to_string();
                let depth = key.split('.').count();
                descendants.push(Node {
                    depth,
                    name,
                    module,
                });
            }
        });
        ModuleDisplay {
            root: module,
            descendants,
        }
    }
}

impl fmt::Display for ModuleDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&repr(self.root, &self.descendants))
    }
}

impl fmt::Debug for ModuleDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// The text of `module`, whose submodules are `descendants`, as PyTorch's `Module.__repr__`
/// writes it (and `ModuleList.__repr__` for a `ModuleList`).
fn repr(module: &dyn Module, descendants: &[Node<'_>]) -> String {
    let mut children = vec![];
    let mut rest = descendants;
    while let Some((child, tail)) = rest.split_first() {
        let end = tail
            .iter()
            .position(|n| n.depth <= child.depth)
            .unwrap_or(tail.len());
        children.push((child.name.as_str(), repr(child.module, &tail[..end])));
        rest = &tail[end..];
    }
    let name = module.type_name();
    let extra = module.extra_repr();
    // An empty text has no lines (Python's `"".split("\n")` would have one).
    let extra_lines: Vec<&str> = if extra.is_empty() {
        vec![]
    } else {
        extra.split('\n').collect()
    };
    let child_lines: Vec<String> = if name == "ModuleList" {
        group_repeated(&children)
    } else {
        children
            .iter()
            .map(|(name, text)| add_indent(&format!("({name}): {text}")))
            .collect()
    };
    let mut out = format!("{name}(");
    if extra_lines.len() == 1 && child_lines.is_empty() {
        out.push_str(extra_lines[0]);
    } else if !extra_lines.is_empty() || !child_lines.is_empty() {
        let lines: Vec<&str> = extra_lines
            .iter()
            .copied()
            .chain(child_lines.iter().map(String::as_str))
            .collect();
        out.push_str("\n  ");
        out.push_str(&lines.join("\n  "));
        out.push('\n');
    }
    out.push(')');
    out
}

/// The lines of the children of a `ModuleList`: a run of children with the same text is one line,
/// `(first-last): n x text`.
fn group_repeated(children: &[(&str, String)]) -> Vec<String> {
    let mut lines = vec![];
    let mut rest = children;
    while let Some(((first, text), tail)) = rest.split_first() {
        let run = tail.iter().take_while(|(_, t)| t == text).count();
        let line = match run {
            0 => format!("({first}): {text}"),
            _ => format!("({first}-{}): {} x {text}", tail[run - 1].0, run + 1),
        };
        lines.push(add_indent(&line));
        rest = &tail[run..];
    }
    lines
}

/// PyTorch's `_addindent(s, 2)`: the lines after the first indented by two spaces, except blank
/// ones, which become empty.
fn add_indent(s: &str) -> String {
    let mut lines = s.split('\n');
    let first = lines.next().unwrap_or_default();
    let mut out = first.to_string();
    for line in lines {
        out.push('\n');
        if !line.trim().is_empty() {
            out.push_str("  ");
            out.push_str(line);
        }
    }
    out
}

/// The name of a type without its path and generic arguments: `Block` for
/// `my_crate::Block<other::Linear>`.
pub(super) fn short_type_name(full: &'static str) -> &'static str {
    let path = full.split('<').next().unwrap_or(full);
    path.rsplit("::").next().unwrap_or(path)
}

/// Python's `repr` of a float: the shortest text that reads back as `x`, with a decimal point,
/// in exponent form (`1e-05`, `1e+16`) below `1e-4` and from `1e16`.
pub(crate) fn python_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    // Rust's exponent form has the shortest digits that read back as `x` (`-1.25e-7`, `0e0`),
    // but when two of that length are equally close to `x`, it doesn't choose as Python does
    // (the even last digit: `1000000000000000.2` for `1000000000000000.25`). `x` rounded to as
    // many digits, half to even, is Python's choice whenever it reads back as `x`; otherwise
    // there is no tie, and Rust's digits are the closest that read back.
    let shortest = format!("{x:e}");
    let length = shortest
        .split_once('e')
        .map_or(0, |(m, _)| m.bytes().filter(u8::is_ascii_digit).count());
    let nearest = format!("{x:.*e}", length.saturating_sub(1));
    let scientific = if nearest.parse::<f64>() == Ok(x) {
        nearest
    } else {
        shortest
    };
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("an exponent in the `{:e}` form");
    let exponent: i32 = exponent.parse().expect("an integer exponent");
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits: String = mantissa.chars().filter(|&c| c != '.').collect();
    let body = if (-4..16).contains(&exponent) {
        if exponent < 0 {
            format!("0.{}{digits}", "0".repeat((-exponent - 1) as usize))
        } else {
            let integer = exponent as usize + 1;
            if digits.len() <= integer {
                format!("{digits}{}.0", "0".repeat(integer - digits.len()))
            } else {
                format!("{}.{}", &digits[..integer], &digits[integer..])
            }
        }
    } else {
        let (first, rest) = digits.split_at(1);
        let mantissa = if rest.is_empty() {
            first.to_string()
        } else {
            format!("{first}.{rest}")
        };
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exponent.abs())
    };
    format!("{sign}{body}")
}

/// Python's `True` / `False`.
pub(crate) fn python_bool(b: bool) -> &'static str {
    if b { "True" } else { "False" }
}

/// An optional float as Python writes it: `None` or the float.
pub(crate) fn python_option_float(x: Option<f64>) -> String {
    x.map_or_else(|| "None".into(), python_float)
}

/// A Python tuple of sizes: `()`, `(4,)`, `(2, 3)`.
pub(crate) fn python_tuple(values: &[usize]) -> String {
    match values {
        [one] => format!("({one},)"),
        _ => {
            let items: Vec<String> = values.iter().map(usize::to_string).collect();
            format!("({})", items.join(", "))
        }
    }
}

/// A size of a pooling layer: one number when the height and the width are equal, as PyTorch
/// prints the number it was given (`kernel_size=2`), or the pair (`(3, 2)`).
pub(crate) fn python_size([h, w]: [usize; 2]) -> String {
    if h == w {
        h.to_string()
    } else {
        python_tuple(&[h, w])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    // The ties are written exactly: rounded, they would read back as the same value.
    #[allow(clippy::excessive_precision)]
    fn floats_are_written_as_python_writes_them() {
        // Python's `repr` of each value.
        let cases = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.5, "0.5"),
            (0.1, "0.1"),
            (1e-5, "1e-05"),
            (1e-4, "0.0001"),
            (1.5e-4, "0.00015"),
            (0.001, "0.001"),
            (1e-20, "1e-20"),
            (1.25e-7, "1.25e-07"),
            (123456.0, "123456.0"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e300, "1.5e+300"),
            (1.0 / 3.0, "0.3333333333333333"),
            (-2.5, "-2.5"),
            (f64::MAX, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
            // Two shortest candidates equally close: the even last digit (R111), up or down.
            (1_000_000_000_000_000.25, "1000000000000000.2"),
            (-1_000_000_000_000_000.25, "-1000000000000000.2"),
            (1_000_000_000_000_000.75, "1000000000000000.8"),
            (252_158_283_539_364.625, "252158283539364.62"),
            (2f64.powi(-25), "2.9802322387695312e-08"),
            (3.0 * 2f64.powi(-24), "1.7881393432617188e-07"),
            // Their neighbours have no tie.
            (1_000_000_000_000_000.25f64.next_up(), "1000000000000000.4"),
            (
                1_000_000_000_000_000.25f64.next_down(),
                "1000000000000000.1",
            ),
            // `2**-1007`, whose nearest decimal of the shortest length doesn't read back: the
            // gap below a power of two is half the gap above.
            (f64::from_bits(16 << 52), "7.291122019556398e-304"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
        ];
        for (x, text) in cases {
            assert_eq!(python_float(x), text, "{x:e}");
        }
    }

    #[test]
    fn type_names_drop_paths_and_generic_arguments() {
        assert_eq!(short_type_name("a::Block<b::Vec<f32>>"), "Block");
        assert_eq!(short_type_name("a::b::Pair<x::Y<z::W>, u8>"), "Pair");
        assert_eq!(short_type_name("Plain"), "Plain");
    }

    #[test]
    fn tuples_and_sizes() {
        assert_eq!(python_tuple(&[]), "()");
        assert_eq!(python_tuple(&[4]), "(4,)");
        assert_eq!(python_tuple(&[2, 3]), "(2, 3)");
        assert_eq!(python_size([2, 2]), "2");
        assert_eq!(python_size([3, 2]), "(3, 2)");
    }

    #[test]
    fn indentation_keeps_blank_lines_empty() {
        assert_eq!(add_indent("a"), "a");
        assert_eq!(add_indent("a(\n  x\n\n  \n)"), "a(\n    x\n\n\n  )");
    }
}
