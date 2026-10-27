//! The ResNet example of the workspace's README isn't compiled there (it is a test of `homura`,
//! which can't depend on homura-models); it must be the example of homura-models' README, which
//! is compiled as a documentation test of this crate.

/// The code blocks of `markdown` whose info string is `info`.
fn blocks(markdown: &str, info: &str) -> Vec<String> {
    let mut out = vec![];
    let mut current: Option<String> = None;
    for line in markdown.lines() {
        match &mut current {
            None if line == format!("```{info}") => current = Some(String::new()),
            Some(_) if line == "```" => out.push(current.take().unwrap()),
            Some(block) => {
                block.push_str(line);
                block.push('\n');
            }
            None => {}
        }
    }
    out
}

#[test]
fn the_workspace_readme_shows_the_example_of_this_crate() {
    let root = include_str!("../../README.md");
    let ours = include_str!("../README.md");
    let ignored: Vec<String> = blocks(root, "rust,ignore")
        .into_iter()
        .filter(|b| b.contains("homura_models"))
        .collect();
    assert_eq!(ignored.len(), 1, "the README has one homura-models example");
    assert!(
        blocks(ours, "rust,no_run").contains(&ignored[0]),
        "the example of README.md differs from homura-models/README.md's:\n{}",
        ignored[0]
    );
}
