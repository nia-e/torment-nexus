//! Versioned extraction settings. Missing settings mean the historical method;
//! only new creation requests default to paper-style extraction.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const LEGACY_ALGORITHM: &str = "last-assistant-content-token-difference-of-means-v1";
pub const PAPER_ALGORITHM: &str = "raw-readout-control-pca-grouped-cv-v1";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    #[default]
    Paper,
    Completion,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Extraction {
    pub method: Method,
    pub readout_suffix: String,
}
impl Default for Extraction {
    fn default() -> Self {
        Self {
            method: Method::Paper,
            readout_suffix: "I feel:".into(),
        }
    }
}
impl Extraction {
    pub fn legacy() -> Self {
        Self {
            method: Method::Completion,
            readout_suffix: String::new(),
        }
    }
    pub fn from_record(record: &Value) -> Result<Self> {
        match record.get("extraction").filter(|v| !v.is_null()) {
            None => Ok(Self::legacy()),
            Some(value) => {
                let settings: Self = serde_json::from_value(value.clone())?;
                ensure!(
                    settings.readout_suffix.len() <= 256,
                    "readout suffix exceeds 256 bytes"
                );
                if settings.method == Method::Paper {
                    ensure!(
                        !settings.readout_suffix.trim().is_empty(),
                        "paper extraction needs a fixed readout suffix"
                    );
                }
                Ok(settings)
            }
        }
    }
    pub fn algorithm(&self) -> &'static str {
        match self.method {
            Method::Paper => PAPER_ALGORITHM,
            Method::Completion => LEGACY_ALGORITHM,
        }
    }
    pub fn capture(&self, pair: &Value, pole: &str, legacy_raw: bool) -> Result<Value> {
        let text = pair[pole].as_str().context("missing pole text")?;
        if self.method == Method::Completion {
            return Ok(json!({"messages":pair["messages"],"completion":text,"raw":legacy_raw}));
        }
        let rendered = format!("{} {}", text.trim_end(), self.readout_suffix);
        let last = rendered.chars().last().context("empty readout")?;
        let cut = rendered.len() - last.len_utf8();
        Ok(
            json!({"messages":[{"role":"user","content":&rendered[..cut]}],"completion":last.to_string(),"raw":true}),
        )
    }
}

pub fn preview_percentages(record: &Value) -> Result<Vec<f64>> {
    let mode = record
        .get("preview_mode")
        .filter(|v| !v.is_null())
        .map(|v| v.as_str().context("preview mode must be text"))
        .transpose()?
        .unwrap_or("standard");
    match mode {
        "none" => Ok(vec![]),
        "negative_only" => Ok(vec![0., -1., -2.]),
        "standard" => Ok(vec![-10., 0., 10.]),
        _ => anyhow::bail!("unknown preview mode"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn old_recipes_do_not_silently_change_method() {
        assert_eq!(
            Extraction::from_record(&json!({})).unwrap().method,
            Method::Completion
        );
        assert_eq!(Extraction::default().method, Method::Paper);
    }
    #[test]
    fn fixed_readout_ignores_chat_context_but_retains_exact_suffix() {
        let x = Extraction::default().capture(&json!({"positive":"A short scene.","messages":[{"role":"user","content":"Not in raw input"}]}), "positive", false).unwrap();
        assert_eq!(x["messages"][0]["content"], "A short scene. I feel");
        assert_eq!(x["completion"], ":");
        assert_eq!(x["raw"], true);
    }
    #[test]
    fn preview_sets_are_explicit_not_a_coefficient_policy() {
        assert_eq!(
            preview_percentages(&json!({"preview_mode":"none"})).unwrap(),
            Vec::<f64>::new()
        );
        assert_eq!(
            preview_percentages(&json!({"preview_mode":"negative_only"})).unwrap(),
            [0., -1., -2.]
        );
        assert_eq!(
            preview_percentages(
                &json!({"coefficient_policy":"non_positive","preview_mode":"standard"})
            )
            .unwrap(),
            [-10., 0., 10.]
        );
        // Historical recipes retain their original preview set.
        assert_eq!(preview_percentages(&json!({})).unwrap(), [-10., 0., 10.]);
    }
}
