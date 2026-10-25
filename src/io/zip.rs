//! Reading zip archives whose entries are stored, not compressed: PyTorch's `.pt` format.
//! Compressed entries can be listed (TorchScript archives have some) but not read.
//!
//! The central directory is the authority on each entry's name, size and CRC. Every local file
//! header must agree with it, and so must the data descriptor that follows the data of entries
//! with general purpose bit 3 set (PyTorch sets it on every entry and leaves the CRC and sizes in
//! the local header zero). ZIP64 is supported. Entries can't overlap each other or the central
//! directory, and each one has a byte range in the file, so callers can read or map the data.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use crate::{Error, Result};

const LOCAL_SIG: u32 = 0x0403_4b50;
const CENTRAL_SIG: u32 = 0x0201_4b50;
const DESCRIPTOR_SIG: u32 = 0x0807_4b50;
const EOCD_SIG: u32 = 0x0605_4b50;
const EOCD64_SIG: u32 = 0x0606_4b50;
const LOCATOR_SIG: u32 = 0x0706_4b50;

const LOCAL_LEN: u64 = 30;
const CENTRAL_LEN: usize = 46;
const EOCD_LEN: usize = 22;
const EOCD64_LEN: u64 = 56;
const LOCATOR_LEN: usize = 20;
const MAX_COMMENT: usize = 0xffff;

/// The extra field that holds ZIP64 sizes and offsets.
const ZIP64_EXTRA: u16 = 0x0001;
/// Encrypted (bit 0), strong encryption (bit 6) and masked headers (bit 13).
const ENCRYPTED: u16 = 0x0001 | 0x0040 | 0x2000;
/// The CRC and sizes follow the data in a data descriptor.
const DESCRIPTOR: u16 = 0x0008;

/// One entry of an archive, checked against its local header and data descriptor.
#[derive(Clone, Debug)]
pub(super) struct Entry {
    pub(super) name: String,
    /// Where the entry's data starts in the file.
    pub(super) start: u64,
    /// The size of the data in the file.
    pub(super) size: u64,
    /// The compression method; only 0 (stored) can be read.
    pub(super) method: u16,
    /// The CRC-32 of the data as the central directory records it; 0 if the writer didn't
    /// compute it.
    pub(super) crc: u32,
}

/// The entries of a zip archive, in central directory order.
#[derive(Debug)]
pub(super) struct Archive {
    entries: Vec<Entry>,
    index: HashMap<String, usize>,
}

fn bad(msg: &str) -> Error {
    Error::Format(format!("invalid zip archive: {msg}"))
}

/// Little-endian fields from a byte slice, with every read bounds-checked.
struct Fields<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Fields<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Fields { bytes, pos: 0 }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or_else(|| bad("a record is truncated"))?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn rest(&self) -> usize {
        self.bytes.len() - self.pos
    }
}

fn read_at<R: Read + Seek>(r: &mut R, offset: u64, len: usize) -> Result<Vec<u8>> {
    r.seek(SeekFrom::Start(offset))?;
    let mut buf = vec![0; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// The values of a ZIP64 extra field: each one replaces the 32-bit (or, for the disk, 16-bit)
/// field that holds the placeholder `0xffffffff`, in this order.
#[derive(Clone, Copy)]
struct Sizes {
    size: u64,
    compressed: u64,
    offset: u64,
    disk: u32,
}

/// Applies the ZIP64 extra field of `extra`, if any, to the fields that hold placeholders.
/// Returns whether there was one.
fn apply_zip64(extra: &[u8], sizes: &mut Sizes, placeholders: [bool; 4]) -> Result<bool> {
    let mut fields = Fields::new(extra);
    let mut found = false;
    while fields.rest() > 0 {
        let id = fields.u16()?;
        let len = fields.u16()? as usize;
        let data = fields.take(len)?;
        if id != ZIP64_EXTRA {
            continue;
        }
        if found {
            return Err(bad("two ZIP64 extra fields"));
        }
        found = true;
        let mut data = Fields::new(data);
        if placeholders[0] {
            sizes.size = data.u64()?;
        }
        if placeholders[1] {
            sizes.compressed = data.u64()?;
        }
        if placeholders[2] {
            sizes.offset = data.u64()?;
        }
        if placeholders[3] {
            sizes.disk = data.u32()?;
        }
    }
    Ok(found)
}

/// The end of central directory record, with ZIP64 values where the archive has them.
struct End {
    entries: u64,
    cd_offset: u64,
    cd_size: u64,
    /// Where the records after the central directory start.
    cd_limit: u64,
}

/// Finds the end of central directory record: the last signature whose comment length reaches
/// exactly the end of the file.
fn read_end<R: Read + Seek>(r: &mut R, len: u64) -> Result<End> {
    if len < EOCD_LEN as u64 {
        return Err(bad("the file is too short"));
    }
    // The record, the longest comment and the ZIP64 locator that precedes the record.
    let tail_len = len.min((EOCD_LEN + MAX_COMMENT + LOCATOR_LEN) as u64) as usize;
    let tail_start = len - tail_len as u64;
    let tail = read_at(r, tail_start, tail_len)?;
    let pos = (0..=tail_len - EOCD_LEN)
        .rev()
        .find(|&p| {
            u32::from_le_bytes(tail[p..p + 4].try_into().unwrap()) == EOCD_SIG
                && u16::from_le_bytes(tail[p + 20..p + 22].try_into().unwrap()) as usize
                    == tail_len - p - EOCD_LEN
        })
        .ok_or_else(|| bad("no end of central directory record"))?;
    let mut f = Fields::new(&tail[pos + 4..pos + EOCD_LEN]);
    let disk = f.u16()?;
    let cd_disk = f.u16()?;
    let disk_entries = f.u16()?;
    let entries = f.u16()?;
    let cd_size = f.u32()?;
    let cd_offset = f.u32()?;
    let eocd = tail_start + pos as u64;

    let locator = pos
        .checked_sub(LOCATOR_LEN)
        .filter(|&l| u32::from_le_bytes(tail[l..l + 4].try_into().unwrap()) == LOCATOR_SIG);
    let Some(locator) = locator else {
        if disk != 0 || cd_disk != 0 || disk_entries != entries {
            return Err(bad("multi-disk archives are not supported"));
        }
        return Ok(End {
            entries: entries.into(),
            cd_offset: cd_offset.into(),
            cd_size: cd_size.into(),
            cd_limit: eocd,
        });
    };
    let mut f = Fields::new(&tail[locator + 4..locator + LOCATOR_LEN]);
    let eocd64_disk = f.u32()?;
    let eocd64 = f.u64()?;
    let disks = f.u32()?;
    let locator = tail_start + locator as u64;
    if eocd64_disk != 0 || disks > 1 {
        return Err(bad("multi-disk archives are not supported"));
    }
    if eocd64
        .checked_add(EOCD64_LEN)
        .is_none_or(|end| end > locator)
    {
        return Err(bad(
            "the ZIP64 end of central directory record is out of range",
        ));
    }
    let record = read_at(r, eocd64, EOCD64_LEN as usize)?;
    let mut f = Fields::new(&record);
    let sig = f.u32()?;
    let record_len = f.u64()?;
    let _made_by = f.u16()?;
    let _needed = f.u16()?;
    let disk64 = f.u32()?;
    let cd_disk64 = f.u32()?;
    let disk_entries64 = f.u64()?;
    let entries64 = f.u64()?;
    let cd_size64 = f.u64()?;
    let cd_offset64 = f.u64()?;
    if sig != EOCD64_SIG || record_len < EOCD64_LEN - 12 {
        return Err(bad("invalid ZIP64 end of central directory record"));
    }
    // The size counts the record without its first 12 bytes; any 64-bit value can be in it.
    if record_len
        .checked_add(12)
        .and_then(|len| eocd64.checked_add(len))
        .is_none_or(|end| end > locator)
    {
        return Err(bad(
            "the ZIP64 end of central directory record is out of range",
        ));
    }
    if disk64 != 0 || cd_disk64 != 0 || disk_entries64 != entries64 {
        return Err(bad("multi-disk archives are not supported"));
    }
    // Fields of the 32-bit record that aren't placeholders must agree with the ZIP64 ones.
    let agree =
        |value: u64, placeholder: u64, value64: u64| value == placeholder || value == value64;
    if !agree(entries.into(), 0xffff, entries64)
        || !agree(disk_entries.into(), 0xffff, disk_entries64)
        || !agree(cd_size.into(), 0xffff_ffff, cd_size64)
        || !agree(cd_offset.into(), 0xffff_ffff, cd_offset64)
    {
        return Err(bad(
            "the end of central directory records disagree with each other",
        ));
    }
    Ok(End {
        entries: entries64,
        cd_offset: cd_offset64,
        cd_size: cd_size64,
        cd_limit: eocd64,
    })
}

/// An entry as the central directory records it.
struct Central {
    name: String,
    flags: u16,
    method: u16,
    crc: u32,
    /// The size in the file.
    compressed: u64,
    size: u64,
    offset: u64,
    /// Whether it has a ZIP64 extra field.
    zip64: bool,
}

fn read_central(f: &mut Fields<'_>) -> Result<Central> {
    if f.u32()? != CENTRAL_SIG {
        return Err(bad("invalid central directory header"));
    }
    let _made_by = f.u16()?;
    let _needed = f.u16()?;
    let flags = f.u16()?;
    let method = f.u16()?;
    let _time = f.u16()?;
    let _date = f.u16()?;
    let crc = f.u32()?;
    let compressed = f.u32()?;
    let size = f.u32()?;
    let name_len = f.u16()? as usize;
    let extra_len = f.u16()? as usize;
    let comment_len = f.u16()? as usize;
    let disk = f.u16()?;
    let _internal = f.u16()?;
    let _external = f.u32()?;
    let offset = f.u32()?;
    let name = f.take(name_len)?;
    let extra = f.take(extra_len)?;
    f.take(comment_len)?;

    let name = std::str::from_utf8(name)
        .map_err(|_| bad("an entry name is not UTF-8"))?
        .to_string();
    if flags & ENCRYPTED != 0 {
        return Err(bad(&format!("{name:?} is encrypted")));
    }
    let mut sizes = Sizes {
        size: size.into(),
        compressed: compressed.into(),
        offset: offset.into(),
        disk: disk.into(),
    };
    let zip64 = apply_zip64(
        extra,
        &mut sizes,
        [
            size == 0xffff_ffff,
            compressed == 0xffff_ffff,
            offset == 0xffff_ffff,
            disk == 0xffff,
        ],
    )?;
    if sizes.disk != 0 {
        return Err(bad("multi-disk archives are not supported"));
    }
    if method == 0 && sizes.compressed != sizes.size {
        return Err(bad(&format!(
            "{name:?} is stored but its compressed and uncompressed sizes differ"
        )));
    }
    Ok(Central {
        name,
        flags,
        method,
        crc,
        compressed: sizes.compressed,
        size: sizes.size,
        offset: sizes.offset,
        zip64,
    })
}

/// Checks the local header of `c`, which must end, with its data and data descriptor, by
/// `limit`, and returns where its data starts.
fn check_local<R: Read + Seek>(r: &mut R, c: &Central, limit: u64) -> Result<u64> {
    let name = &c.name;
    let header_end = c
        .offset
        .checked_add(LOCAL_LEN)
        .filter(|&end| end <= limit)
        .ok_or_else(|| bad(&format!("the local header of {name:?} is out of range")))?;
    let header = read_at(r, c.offset, LOCAL_LEN as usize)?;
    let mut f = Fields::new(&header);
    let sig = f.u32()?;
    let _needed = f.u16()?;
    let flags = f.u16()?;
    let method = f.u16()?;
    let _time = f.u16()?;
    let _date = f.u16()?;
    let crc = f.u32()?;
    let compressed = f.u32()?;
    let size = f.u32()?;
    let name_len = f.u16()? as u64;
    let extra_len = f.u16()? as u64;
    if sig != LOCAL_SIG {
        return Err(bad(&format!("invalid local header for {name:?}")));
    }
    let start = header_end + name_len + extra_len;
    if start > limit {
        return Err(bad(&format!(
            "the local header of {name:?} is out of range"
        )));
    }
    let rest = read_at(r, header_end, (name_len + extra_len) as usize)?;
    let (local_name, extra) = rest.split_at(name_len as usize);
    if local_name != name.as_bytes() {
        return Err(bad(&format!(
            "the local header of {name:?} has another name"
        )));
    }
    if method != c.method || flags & DESCRIPTOR != c.flags & DESCRIPTOR {
        return Err(bad(&format!(
            "the local header of {name:?} disagrees with the central directory"
        )));
    }
    let mut sizes = Sizes {
        size: size.into(),
        compressed: compressed.into(),
        offset: 0,
        disk: 0,
    };
    let zip64 = apply_zip64(
        extra,
        &mut sizes,
        [size == 0xffff_ffff, compressed == 0xffff_ffff, false, false],
    )?;
    let end = start
        .checked_add(c.compressed)
        .filter(|&end| end <= limit)
        .ok_or_else(|| bad(&format!("the data of {name:?} is out of range")))?;
    if flags & DESCRIPTOR == 0 {
        if crc != c.crc || sizes.size != c.size || sizes.compressed != c.compressed {
            return Err(bad(&format!(
                "the local header of {name:?} disagrees with the central directory"
            )));
        }
    } else {
        // The CRC and sizes of the local header are placeholders; the data descriptor after the
        // data has the real values. Its sizes are 64-bit for ZIP64 entries.
        check_descriptor(r, c, end, limit, c.zip64 || zip64)?;
    }
    Ok(start)
}

/// Checks the data descriptor of `c` at `at`: an optional signature, the CRC and the two sizes,
/// which must equal those of the central directory and end by `limit`. Writers disagree on
/// when the sizes are 64-bit, so the other width is accepted too if it matches.
fn check_descriptor<R: Read + Seek>(
    r: &mut R,
    c: &Central,
    at: u64,
    limit: u64,
    zip64: bool,
) -> Result<()> {
    let available = (limit - at).min(24) as usize;
    let bytes = read_at(r, at, available)?;
    let matches = |wide: bool| -> bool {
        let mut f = Fields::new(&bytes);
        let parse = |f: &mut Fields<'_>| -> Result<bool> {
            let crc = f.u32()?;
            let (compressed, size) = if wide {
                (f.u64()?, f.u64()?)
            } else {
                (f.u32()?.into(), f.u32()?.into())
            };
            Ok(crc == c.crc && compressed == c.compressed && size == c.size)
        };
        let signed = bytes.len() >= 4
            && u32::from_le_bytes(bytes[..4].try_into().unwrap()) == DESCRIPTOR_SIG;
        if signed {
            let mut g = Fields::new(&bytes[4..]);
            if parse(&mut g).unwrap_or(false) {
                return true;
            }
        }
        // A CRC that happens to equal the signature is possible without one.
        parse(&mut f).unwrap_or(false)
    };
    if matches(zip64) || matches(!zip64) {
        Ok(())
    } else {
        Err(bad(&format!(
            "the data descriptor of {:?} is missing or disagrees with the central directory",
            c.name
        )))
    }
}

impl Archive {
    /// Reads and checks the central directory and every local header of the archive in `r`.
    pub(super) fn read<R: Read + Seek>(r: &mut R) -> Result<Archive> {
        let len = r.seek(SeekFrom::End(0))?;
        let end = read_end(r, len)?;
        if end
            .cd_offset
            .checked_add(end.cd_size)
            .is_none_or(|e| e > end.cd_limit)
        {
            return Err(bad("the central directory is out of range"));
        }
        // Each header takes at least 46 bytes, which bounds the number of entries by the file size.
        if end.entries > end.cd_size / CENTRAL_LEN as u64 {
            return Err(bad("the central directory is too small for its entries"));
        }
        // Within the file, so allocating it is bounded by the file size.
        let cd_size =
            usize::try_from(end.cd_size).map_err(|_| bad("the central directory is too large"))?;
        let directory = read_at(r, end.cd_offset, cd_size)?;
        let mut f = Fields::new(&directory);
        let mut centrals = Vec::with_capacity(end.entries as usize);
        let mut index = HashMap::with_capacity(end.entries as usize);
        for i in 0..end.entries as usize {
            let c = read_central(&mut f)?;
            if index.insert(c.name.clone(), i).is_some() {
                return Err(bad(&format!("{:?} appears twice", c.name)));
            }
            centrals.push(c);
        }
        if f.rest() != 0 {
            return Err(bad("the central directory has more entries than recorded"));
        }

        // Entries in file order: each one must end before the next one starts.
        let mut order: Vec<usize> = (0..centrals.len()).collect();
        order.sort_unstable_by_key(|&i| centrals[i].offset);
        let mut starts = vec![0; centrals.len()];
        for (k, &i) in order.iter().enumerate() {
            let limit = order
                .get(k + 1)
                .map_or(end.cd_offset, |&next| centrals[next].offset);
            starts[i] = check_local(r, &centrals[i], limit)?;
        }
        let entries = centrals
            .into_iter()
            .zip(starts)
            .map(|(c, start)| Entry {
                name: c.name,
                start,
                size: c.compressed,
                method: c.method,
                crc: c.crc,
            })
            .collect();
        Ok(Archive { entries, index })
    }

    /// The entries, in central directory order.
    pub(super) fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub(super) fn get(&self, name: &str) -> Option<&Entry> {
        self.index.get(name).map(|&i| &self.entries[i])
    }
}

/// Checks that `entry` can be read: only stored entries can.
pub(super) fn check_stored(entry: &Entry) -> Result<()> {
    if entry.method != 0 {
        return Err(Error::Format(format!(
            "unsupported zip archive: {:?} is compressed (method {}); only stored entries are \
             supported",
            entry.name, entry.method
        )));
    }
    Ok(())
}

/// Reads the data of `entry`, checking its CRC unless the recorded one is 0, which writers use
/// when they don't compute it (PyTorch's `set_crc32_options(False)`).
pub(super) fn read_entry<R: Read + Seek>(r: &mut R, entry: &Entry) -> Result<Vec<u8>> {
    check_stored(entry)?;
    // The archive check put the data inside the file, so its size fits in memory as far as the
    // file does.
    let size = usize::try_from(entry.size).map_err(|_| bad("an entry is too large"))?;
    let mut data = vec![0; size];
    read_entry_into(r, entry, &mut data)?;
    Ok(data)
}

/// Reads the data of `entry` into `buf`, which must have the entry's size, with the CRC check
/// of [`read_entry`].
pub(super) fn read_entry_into<R: Read + Seek>(
    r: &mut R,
    entry: &Entry,
    buf: &mut [u8],
) -> Result<()> {
    check_stored(entry)?;
    assert_eq!(
        buf.len() as u64,
        entry.size,
        "the buffer has the entry's size"
    );
    r.seek(SeekFrom::Start(entry.start))?;
    r.read_exact(buf)?;
    if entry.crc != 0 && crc32(buf) != entry.crc {
        return Err(bad(&format!(
            "the CRC of {:?} doesn't match its data",
            entry.name
        )));
    }
    Ok(())
}

const CRC_TABLE: [u32; 256] = {
    let mut table = [0; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

/// The CRC-32 of zip archives (ISO-HDLC, reflected polynomial `0xedb88320`).
pub(super) fn crc32(data: &[u8]) -> u32 {
    !data.iter().fold(!0u32, |c, &b| {
        CRC_TABLE[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8)
    })
}

/// Writes zip archives with the layouts that the readers must handle, for tests.
#[cfg(test)]
pub(super) mod testing {
    use super::crc32;

    /// How [`Writer::add`] writes an entry.
    #[derive(Clone, Copy, Default)]
    pub(crate) struct Options {
        /// Set bit 3 and write a data descriptor (with a signature) after the data.
        pub(crate) descriptor: bool,
        /// Write ZIP64 sizes in the local header (and the descriptor) and ZIP64 sizes and offset
        /// in the central directory.
        pub(crate) zip64: bool,
        /// The CRC to record instead of the real one.
        pub(crate) crc: Option<u32>,
    }

    #[derive(Default)]
    pub(crate) struct Writer {
        pub(crate) out: Vec<u8>,
        central: Vec<u8>,
        count: u16,
    }

    fn put16(v: &mut Vec<u8>, x: u16) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    fn put32(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    fn put64(v: &mut Vec<u8>, x: u64) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    impl Writer {
        /// Adds a stored entry and returns the offset of its local header.
        pub(crate) fn add(&mut self, name: &str, data: &[u8], o: Options) -> usize {
            let crc = o.crc.unwrap_or_else(|| crc32(data));
            let size = data.len() as u32;
            let flags = 0x800 | if o.descriptor { 0x08 } else { 0 };
            let offset = self.out.len();
            let v = &mut self.out;
            put32(v, super::LOCAL_SIG);
            put16(v, if o.zip64 { 45 } else { 20 });
            put16(v, flags);
            put16(v, 0);
            put32(v, 0);
            put32(v, if o.descriptor { 0 } else { crc });
            let local_size = match (o.zip64, o.descriptor) {
                (true, _) => 0xffff_ffff,
                (false, true) => 0,
                (false, false) => size,
            };
            put32(v, local_size);
            put32(v, local_size);
            put16(v, name.len() as u16);
            put16(v, if o.zip64 { 20 } else { 0 });
            v.extend_from_slice(name.as_bytes());
            if o.zip64 {
                put16(v, super::ZIP64_EXTRA);
                put16(v, 16);
                let sizes = if o.descriptor { 0 } else { data.len() as u64 };
                put64(v, sizes);
                put64(v, sizes);
            }
            v.extend_from_slice(data);
            if o.descriptor {
                put32(v, super::DESCRIPTOR_SIG);
                put32(v, crc);
                if o.zip64 {
                    put64(v, data.len() as u64);
                    put64(v, data.len() as u64);
                } else {
                    put32(v, size);
                    put32(v, size);
                }
            }
            let c = &mut self.central;
            put32(c, super::CENTRAL_SIG);
            put16(c, 45);
            put16(c, if o.zip64 { 45 } else { 20 });
            put16(c, flags);
            put16(c, 0);
            put32(c, 0);
            put32(c, crc);
            let central_size = if o.zip64 { 0xffff_ffff } else { size };
            put32(c, central_size);
            put32(c, central_size);
            put16(c, name.len() as u16);
            put16(c, if o.zip64 { 28 } else { 0 });
            put16(c, 0);
            put16(c, 0);
            put16(c, 0);
            put32(c, 0);
            put32(c, if o.zip64 { 0xffff_ffff } else { offset as u32 });
            c.extend_from_slice(name.as_bytes());
            if o.zip64 {
                put16(c, super::ZIP64_EXTRA);
                put16(c, 24);
                put64(c, data.len() as u64);
                put64(c, data.len() as u64);
                put64(c, offset as u64);
            }
            self.count += 1;
            offset
        }

        /// Appends the central directory and the end records (with ZIP64 ones before them if
        /// `zip64`, as PyTorch writes them) and a comment.
        pub(crate) fn finish(mut self, zip64: bool, comment: &[u8]) -> Vec<u8> {
            let cd_offset = self.out.len();
            let cd_size = self.central.len();
            self.out.extend_from_slice(&self.central);
            let v = &mut self.out;
            if zip64 {
                let eocd64 = v.len();
                put32(v, super::EOCD64_SIG);
                put64(v, 44);
                put16(v, 45);
                put16(v, 45);
                put32(v, 0);
                put32(v, 0);
                put64(v, self.count.into());
                put64(v, self.count.into());
                put64(v, cd_size as u64);
                put64(v, cd_offset as u64);
                put32(v, super::LOCATOR_SIG);
                put32(v, 0);
                put64(v, eocd64 as u64);
                put32(v, 1);
            }
            put32(v, super::EOCD_SIG);
            put16(v, 0);
            put16(v, 0);
            put16(v, self.count);
            put16(v, self.count);
            put32(v, cd_size as u32);
            put32(v, cd_offset as u32);
            put16(v, comment.len() as u16);
            v.extend_from_slice(comment);
            self.out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::fixtures::{self, pt};
    use super::testing::{Options, Writer};
    use super::*;
    use std::io::Cursor;

    fn read(bytes: &[u8]) -> Result<Archive> {
        Archive::read(&mut Cursor::new(bytes))
    }

    fn data(bytes: &[u8], archive: &Archive, name: &str) -> Result<Vec<u8>> {
        read_entry(&mut Cursor::new(bytes), archive.get(name).unwrap())
    }

    fn err(result: Result<impl std::fmt::Debug>, message: &str) {
        match result {
            Err(Error::Format(m)) if m.contains(message) => {}
            other => panic!("expected a format error with {message:?}, got {other:?}"),
        }
    }

    /// Two entries in each layout, with a comment.
    fn archive(o: Options, zip64_end: bool) -> Vec<u8> {
        let mut w = Writer::default();
        w.add("a/one", b"first", o);
        w.add("a/two", &[7; 300], o);
        w.finish(zip64_end, b"a comment")
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(b"123456789"), 0xcbf4_3926);
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn torch_save_archives_are_read() {
        if !fixtures::available() {
            return;
        }
        // Every entry has bit 3 and a 16-byte descriptor, the data is 64-byte aligned and the
        // archive ends with ZIP64 records (M3.3).
        for name in ["state_dict.pt", "checkpoint.pt", "values_p2.pt"] {
            let bytes = pt(name);
            let archive = read(&bytes).unwrap();
            let names: Vec<&str> = archive.entries().iter().map(|e| e.name.as_str()).collect();
            assert_eq!(names[0], "archive/data.pkl", "{name}");
            assert!(names.contains(&"archive/version"));
            for e in archive.entries() {
                assert!(
                    e.start.is_multiple_of(64) || e.size == 0,
                    "{name}: {}",
                    e.name
                );
                let d = data(&bytes, &archive, &e.name).unwrap();
                assert_eq!(d.len() as u64, e.size);
                assert_eq!(crc32(&d), e.crc);
            }
            assert_eq!(data(&bytes, &archive, "archive/version").unwrap(), b"3\n");
        }
    }

    #[test]
    fn python_zipfile_archives_are_read() {
        if !fixtures::available() {
            return;
        }
        // Without descriptors and alignment, and with ZIP64 local headers.
        for name in ["repacked.pt", "zip64.pt"] {
            let bytes = pt(name);
            let archive = read(&bytes).unwrap();
            assert_eq!(archive.entries()[0].name, "model_v1/data.pkl");
            for e in archive.entries() {
                data(&bytes, &archive, &e.name).unwrap();
            }
        }
    }

    #[test]
    fn every_layout_is_read() {
        for descriptor in [false, true] {
            for zip64 in [false, true] {
                for zip64_end in [false, true] {
                    let o = Options {
                        descriptor,
                        zip64,
                        crc: None,
                    };
                    let bytes = archive(o, zip64_end);
                    let archive = read(&bytes).unwrap();
                    assert_eq!(data(&bytes, &archive, "a/one").unwrap(), b"first");
                    assert_eq!(data(&bytes, &archive, "a/two").unwrap(), [7; 300]);
                    assert!(archive.get("a/three").is_none());
                }
            }
        }
        // An empty archive, and descriptors without a signature.
        assert!(
            read(&Writer::default().finish(false, b""))
                .unwrap()
                .entries()
                .is_empty()
        );
        let mut w = Writer::default();
        let o = Options {
            descriptor: true,
            ..Options::default()
        };
        w.add("a/x", b"xyz", o);
        let mut bytes = w.finish(false, b"");
        let at = LOCAL_LEN as usize + 3 + 3;
        bytes.drain(at..at + 4);
        // The central directory moved back by 4 bytes.
        let len = bytes.len();
        let offset = u32::from_le_bytes(bytes[len - 6..len - 2].try_into().unwrap()) - 4;
        bytes[len - 6..len - 2].copy_from_slice(&offset.to_le_bytes());
        let archive = read(&bytes).unwrap();
        assert_eq!(data(&bytes, &archive, "a/x").unwrap(), b"xyz");
    }

    #[test]
    fn crcs_are_checked_unless_zero() {
        if !fixtures::available() {
            return;
        }
        let mut bytes = pt("values_p2.pt");
        let archive = read(&bytes).unwrap();
        let start = archive.get("archive/data.pkl").unwrap().start as usize;
        bytes[start + 10] ^= 1;
        let archive = read(&bytes).unwrap();
        err(
            data(&bytes, &archive, "archive/data.pkl"),
            "the CRC of \"archive/data.pkl\" doesn't match",
        );
        // CRC computation was off: corruption can't be noticed.
        let mut bytes = pt("crc0.pt");
        let archive = read(&bytes).unwrap();
        assert!(archive.entries().iter().all(|e| e.crc == 0));
        let start = archive.get("archive/data.pkl").unwrap().start as usize;
        bytes[start + 10] ^= 1;
        data(&bytes, &archive, "archive/data.pkl").unwrap();
        // A wrong CRC in the writer's records.
        let o = Options {
            crc: Some(12345),
            ..Options::default()
        };
        let bytes = archive_with(o);
        let archive = read(&bytes).unwrap();
        err(data(&bytes, &archive, "a/one"), "CRC");
    }

    fn archive_with(o: Options) -> Vec<u8> {
        archive(o, false)
    }

    /// Changes the bytes of an archive.
    type Patch<'a> = dyn Fn(&mut Vec<u8>) + 'a;

    fn patch32(bytes: &mut [u8], at: usize, value: u32) {
        bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn patch16(bytes: &mut [u8], at: usize, value: u16) {
        bytes[at..at + 2].copy_from_slice(&value.to_le_bytes());
    }

    /// The offset of the central directory header of entry `i` of [`archive`].
    fn central(bytes: &[u8], i: usize) -> usize {
        let mut at = bytes
            .windows(4)
            .position(|w| w == CENTRAL_SIG.to_le_bytes())
            .unwrap();
        for _ in 0..i {
            let n = u16::from_le_bytes(bytes[at + 28..at + 30].try_into().unwrap()) as usize;
            let e = u16::from_le_bytes(bytes[at + 30..at + 32].try_into().unwrap()) as usize;
            at += CENTRAL_LEN + n + e;
        }
        at
    }

    #[test]
    fn compressed_entries_are_listed_but_not_read() {
        let mut bytes = archive_with(Options::default());
        let c = central(&bytes, 0);
        patch16(&mut bytes, 8, 8);
        patch16(&mut bytes, c + 10, 8);
        let archive = read(&bytes).unwrap();
        err(data(&bytes, &archive, "a/one"), "is compressed (method 8)");
        assert_eq!(data(&bytes, &archive, "a/two").unwrap(), [7; 300]);
        // Its compressed size can differ from its size.
        patch32(&mut bytes, 18, 3);
        patch32(&mut bytes, c + 20, 3);
        let archive = read(&bytes).unwrap();
        assert_eq!(archive.get("a/one").unwrap().size, 3);
        // A stored entry's can't.
        let mut bytes = archive_with(Options::default());
        let c = central(&bytes, 0);
        patch32(&mut bytes, c + 20, 3);
        err(read(&bytes), "sizes differ");
    }

    #[test]
    fn data_descriptors_must_match_the_central_directory() {
        let o = Options {
            descriptor: true,
            ..Options::default()
        };
        let good = archive_with(o);
        read(&good).unwrap();
        // The descriptor of "a/one" follows its 5 bytes of data.
        let descriptor = LOCAL_LEN as usize + 5 + 5;
        // The CRC, the compressed size and the size.
        for field in [4, 8, 12] {
            let mut bytes = good.clone();
            bytes[descriptor + field] ^= 1;
            err(read(&bytes), "the data descriptor of \"a/one\"");
        }
        // Missing: the second entry's header follows the data right away.
        let mut bytes = archive_with(Options::default());
        let c = central(&bytes, 0);
        patch16(&mut bytes, 6, 0x808);
        patch16(&mut bytes, c + 8, 0x808);
        err(read(&bytes), "the data descriptor of \"a/one\" is missing");
        // A descriptor past the central directory.
        let mut w = Writer::default();
        w.add("a/one", b"first", o);
        let mut bytes = w.finish(false, b"");
        let cut = LOCAL_LEN as usize + 5 + 5 + 8;
        bytes.drain(cut..cut + 8);
        let len = bytes.len();
        patch32(&mut bytes, len - 6, cut as u32);
        err(read(&bytes), "data descriptor");
    }

    #[test]
    fn local_headers_must_match_the_central_directory() {
        let good = archive_with(Options::default());
        // The name, sizes and CRC of a local header without bit 3.
        for (at, value, message) in [
            (30, b'b', "has another name"),
            (14, 0, "disagrees with the central directory"),
            (18, 4, "disagrees with the central directory"),
            (22, 4, "disagrees with the central directory"),
        ] {
            let mut bytes = good.clone();
            if at == 30 {
                bytes[at] = value;
            } else {
                patch32(&mut bytes, at, value.into());
            }
            err(read(&bytes), message);
        }
        // Bit 3 in only one of them, either way, and a compressed local header for a stored
        // entry.
        let local = "the local header of \"a/one\" disagrees with the central directory";
        let mut bytes = good.clone();
        patch16(&mut bytes, 6, 0x808);
        err(read(&bytes), local);
        let mut bytes = good.clone();
        let c = central(&bytes, 0);
        patch16(&mut bytes, c + 8, 0x808);
        err(read(&bytes), local);
        let mut bytes = good.clone();
        patch16(&mut bytes, 8, 8);
        err(read(&bytes), "disagrees with the central directory");
        // A local header that isn't one.
        let mut bytes = good.clone();
        bytes[0] = b'X';
        err(read(&bytes), "invalid local header");
    }

    #[test]
    fn central_directory_errors() {
        let good = archive_with(Options::default());
        let c = central(&good, 0);
        let c1 = central(&good, 1);
        let cases: Vec<(Box<Patch<'_>>, &str)> = vec![
            // The local header must have the same method.
            (
                Box::new(|b| patch16(b, c + 10, 8)),
                "disagrees with the central directory",
            ),
            (Box::new(|b| patch16(b, c + 8, 0x801)), "is encrypted"),
            (Box::new(|b| patch32(b, c + 20, 4)), "sizes differ"),
            (
                Box::new(|b| b[c1 + 48..c1 + 51].copy_from_slice(b"one")),
                "appears twice",
            ),
            (Box::new(|b| b[c + 46] = 0xff), "not UTF-8"),
            (Box::new(|b| patch16(b, c + 34, 1)), "multi-disk"),
            (
                Box::new(|b| b[c] = b'X'),
                "invalid central directory header",
            ),
            // Both entries at the same offset: they overlap.
            (Box::new(|b| patch32(b, c1 + 42, 0)), "out of range"),
            // An entry starting inside the central directory.
            (
                Box::new(move |b| patch32(b, c1 + 42, c as u32 - 10)),
                "out of range",
            ),
            // A size that runs into the next entry.
            (
                Box::new(|b| {
                    patch32(b, c + 20, 100);
                    patch32(b, c + 24, 100);
                    patch32(b, 18, 100);
                    patch32(b, 22, 100);
                }),
                "out of range",
            ),
            // A ZIP64 placeholder without the extra field is taken literally.
            (
                Box::new(|b| patch32(b, c + 42, 0xffff_ffff)),
                "out of range",
            ),
        ];
        for (patch, message) in cases {
            let mut bytes = good.clone();
            patch(&mut bytes);
            err(read(&bytes), message);
        }
    }

    #[test]
    fn end_record_errors() {
        let good = archive_with(Options::default());
        let len = good.len();
        let eocd = len - EOCD_LEN - 9;
        // Entry counts and directory ranges that don't fit.
        for (at, value, message) in [
            (eocd + 8, 3u32, "disk"),
            (eocd + 10, 3, "more entries than recorded"),
            (eocd + 12, 10_000, "out of range"),
            (eocd + 16, 1, "invalid central directory header"),
        ] {
            let mut bytes = good.clone();
            if at == eocd + 8 {
                patch16(&mut bytes, at, value as u16);
            } else if at == eocd + 10 {
                patch16(&mut bytes, at - 2, 1);
                patch16(&mut bytes, at, 1);
            } else {
                patch32(&mut bytes, at, value);
            }
            err(read(&bytes), message);
        }
        // More entries than the directory can hold.
        let mut bytes = good.clone();
        patch16(&mut bytes, eocd + 8, 1000);
        patch16(&mut bytes, eocd + 10, 1000);
        err(read(&bytes), "too small for its entries");
        // A comment length that doesn't reach the end, and no record at all.
        let mut bytes = good.clone();
        patch16(&mut bytes, eocd + 20, 8);
        err(read(&bytes), "no end of central directory record");
        err(read(b"PK\x03\x04rest"), "the file is too short");
        err(read(&[0; 100]), "no end of central directory record");
        // The ZIP64 records must agree with the 32-bit one and stay in range.
        let good = archive(Options::default(), true);
        read(&good).unwrap();
        let len = good.len();
        let locator = len - EOCD_LEN - 9 - LOCATOR_LEN;
        let eocd64 = locator - EOCD64_LEN as usize;
        for (patch, message) in [
            (
                Box::new(|b: &mut Vec<u8>| patch32(b, eocd64 + 40, 7)) as Box<Patch<'_>>,
                "disagree with each other",
            ),
            (Box::new(|b| b[eocd64] = b'X'), "invalid ZIP64"),
            (
                Box::new(|b| patch32(b, locator + 8, 1 << 20)),
                "out of range",
            ),
            (Box::new(|b| patch32(b, locator + 16, 2)), "multi-disk"),
            (Box::new(|b| patch32(b, eocd64 + 4, 1000)), "out of range"),
            // Sizes whose end overflows (R115): adding the 12 bytes before the size, or the
            // record's offset.
            (
                Box::new(|b| b[eocd64 + 4..eocd64 + 12].copy_from_slice(&[0xff; 8])),
                "out of range",
            ),
            (
                Box::new(|b| {
                    b[eocd64 + 4..eocd64 + 12].copy_from_slice(&(u64::MAX - 12).to_le_bytes())
                }),
                "out of range",
            ),
        ] {
            let mut bytes = good.clone();
            patch(&mut bytes);
            err(read(&bytes), message);
        }
        // Placeholders in the 32-bit record take the ZIP64 values.
        let mut bytes = good.clone();
        let eocd = len - EOCD_LEN - 9;
        patch16(&mut bytes, eocd + 8, 0xffff);
        patch16(&mut bytes, eocd + 10, 0xffff);
        patch32(&mut bytes, eocd + 12, 0xffff_ffff);
        patch32(&mut bytes, eocd + 16, 0xffff_ffff);
        assert_eq!(read(&bytes).unwrap().entries().len(), 2);
    }

    #[test]
    fn truncated_archives_are_errors() {
        if !fixtures::available() {
            return;
        }
        for good in [pt("state_dict.pt"), pt("zip64.pt")] {
            for len in (0..good.len()).step_by(7) {
                assert!(read(&good[..len]).is_err(), "{len}");
            }
        }
    }
}
