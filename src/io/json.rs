//! A minimal JSON parser and writer, enough for safetensors headers.

use std::collections::HashSet;
use std::fmt::Write;

use crate::{Error, Result};

/// A parsed JSON value. Object members keep their order; numbers keep their source text so that
/// large integers survive.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Number(String),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Json::Number(n) => n.parse().ok(),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[Json]> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    pub fn as_object(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Object(o) => Some(o),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        self.as_object()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
}

/// Parses a complete JSON document.
pub(crate) fn parse(text: &str) -> Result<Json> {
    let mut p = Parser {
        s: text.as_bytes(),
        i: 0,
    };
    let v = p.value(0)?;
    p.ws();
    if p.i != p.s.len() {
        return Err(p.err("trailing characters"));
    }
    Ok(v)
}

const MAX_DEPTH: usize = 128;

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn err(&self, msg: &str) -> Error {
        Error::Format(format!("invalid JSON at byte {}: {msg}", self.i))
    }

    fn ws(&mut self) {
        while self.i < self.s.len() && matches!(self.s[self.i], b' ' | b'\t' | b'\n' | b'\r') {
            self.i += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn expect(&mut self, c: u8) -> Result<()> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            Err(self.err(&format!("expected '{}'", c as char)))
        }
    }

    fn literal(&mut self, word: &str, v: Json) -> Result<Json> {
        if self.s[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Ok(v)
        } else {
            Err(self.err("unexpected token"))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json> {
        if depth > MAX_DEPTH {
            return Err(self.err("nesting too deep"));
        }
        self.ws();
        match self.peek() {
            Some(b'{') => {
                self.i += 1;
                let mut members = vec![];
                let mut seen = HashSet::new();
                self.ws();
                if self.peek() == Some(b'}') {
                    self.i += 1;
                    return Ok(Json::Object(members));
                }
                loop {
                    self.ws();
                    let key_start = self.i;
                    let key = self.string()?;
                    if !seen.insert(key.clone()) {
                        self.i = key_start;
                        return Err(self.err(&format!("duplicate key {key:?}")));
                    }
                    self.ws();
                    self.expect(b':')?;
                    let v = self.value(depth + 1)?;
                    members.push((key, v));
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b'}') => {
                            self.i += 1;
                            return Ok(Json::Object(members));
                        }
                        _ => return Err(self.err("expected ',' or '}'")),
                    }
                }
            }
            Some(b'[') => {
                self.i += 1;
                let mut items = vec![];
                self.ws();
                if self.peek() == Some(b']') {
                    self.i += 1;
                    return Ok(Json::Array(items));
                }
                loop {
                    items.push(self.value(depth + 1)?);
                    self.ws();
                    match self.peek() {
                        Some(b',') => self.i += 1,
                        Some(b']') => {
                            self.i += 1;
                            return Ok(Json::Array(items));
                        }
                        _ => return Err(self.err("expected ',' or ']'")),
                    }
                }
            }
            Some(b'"') => Ok(Json::String(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(c) if c == b'-' || c.is_ascii_digit() => self.number(),
            _ => Err(self.err("unexpected character")),
        }
    }

    /// Skips ASCII digits and returns how many there were.
    fn digits(&mut self) -> usize {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        self.i - start
    }

    /// `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`, as in RFC 8259.
    fn number(&mut self) -> Result<Json> {
        let start = self.i;
        if self.peek() == Some(b'-') {
            self.i += 1;
        }
        match self.peek() {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.err("invalid number")),
        }
        if self.peek() == Some(b'.') {
            self.i += 1;
            if self.digits() == 0 {
                return Err(self.err("invalid number"));
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.i += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.i += 1;
            }
            if self.digits() == 0 {
                return Err(self.err("invalid number"));
            }
        }
        let text = std::str::from_utf8(&self.s[start..self.i]).unwrap();
        Ok(Json::Number(text.to_string()))
    }

    fn hex4(&mut self) -> Result<u32> {
        let digits = self
            .s
            .get(self.i..self.i + 4)
            .ok_or_else(|| self.err("truncated escape"))?;
        // `from_str_radix` alone would also accept a leading `+`.
        if !digits.iter().all(u8::is_ascii_hexdigit) {
            return Err(self.err("invalid escape"));
        }
        let text = std::str::from_utf8(digits).unwrap();
        let v = u32::from_str_radix(text, 16).unwrap();
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        loop {
            let c = self.peek().ok_or_else(|| self.err("unterminated string"))?;
            self.i += 1;
            match c {
                b'"' => break,
                b'\\' => {
                    let e = self.peek().ok_or_else(|| self.err("unterminated string"))?;
                    self.i += 1;
                    let ch = match e {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let mut code = self.hex4()?;
                            if (0xD800..0xDC00).contains(&code) {
                                if !self.s[self.i..].starts_with(b"\\u") {
                                    return Err(self.err("unpaired surrogate"));
                                }
                                self.i += 2;
                                let low = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&low) {
                                    return Err(self.err("unpaired surrogate"));
                                }
                                code = 0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00);
                            }
                            char::from_u32(code).ok_or_else(|| self.err("invalid code point"))?
                        }
                        _ => return Err(self.err("invalid escape")),
                    };
                    let mut buf = [0; 4];
                    out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                }
                c if c < 0x20 => return Err(self.err("control character in string")),
                c => out.push(c),
            }
        }
        String::from_utf8(out).map_err(|_| self.err("invalid UTF-8"))
    }
}

/// Appends `s` as a JSON string literal.
pub(crate) fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_values() {
        let v =
            parse(r#" {"a": [1, -2.5e3, true, null], "b\n\u00e9\ud83d\ude00": {"c": "x\"y"}} "#)
                .unwrap();
        assert_eq!(v.get("a").unwrap().as_array().unwrap()[0].as_u64(), Some(1));
        assert_eq!(
            v.get("a").unwrap().as_array().unwrap()[1],
            Json::Number("-2.5e3".into())
        );
        let (key, inner) = &v.as_object().unwrap()[1];
        assert_eq!(key, "b\né😀");
        assert_eq!(inner.get("c").unwrap().as_str(), Some("x\"y"));
        assert_eq!(
            parse("18446744073709551615").unwrap().as_u64(),
            Some(u64::MAX)
        );
        for ok in ["0", "-0", "0.5", "-1.25e-3", "1E+9", "10"] {
            assert_eq!(parse(ok).unwrap(), Json::Number(ok.into()));
        }
    }

    #[test]
    fn reject_invalid() {
        for bad in [
            "",
            "{",
            "[1,]",
            "{\"a\" 1}",
            "\"abc",
            "tru",
            "1 2",
            "\"\\ud800\"",
            "\"\\u+001\"",
            "\"\\u-001\"",
            "\"\\u00g1\"",
            "\"\\u00e\"",
            "\"\\ud83d\\u+e00\"",
            "01",
            "-",
            "-01",
            "1.",
            ".5",
            "1e",
            "1e+",
            "+1",
            "1.5.2",
            "--1",
            "{\"a\":1,\"a\":2}",
            "[{\"b\":{\"c\":1,\"c\":1}}]",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} should fail");
        }
    }

    #[test]
    fn string_roundtrip() {
        let mut s = String::new();
        write_string(&mut s, "a\"b\\c\n\u{1}é");
        assert_eq!(parse(&s).unwrap().as_str(), Some("a\"b\\c\n\u{1}é"));
    }
}
