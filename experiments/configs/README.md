# Model configurations

The generator and judge configurations include:

- `gpt-4.1.json`: GPT-4.1 generator;
- `gpt-oss-high.json`: GPT-OSS-120B with high reasoning effort;
- `gpt-oss-medium.json`: GPT-OSS-120B with medium reasoning effort;
- `gpt-5.6-luna.json`: GPT-5.6-Luna with reasoning disabled;
- `judge-gpt-6.1-terra.json`: GPT-6.1 Terra evaluation judge with reasoning disabled.

Credentials are read from the environment variables listed in
`../keys.env.example`. Newly generated files are written under `../runs/` and
are not part of the released paper results.

`gpt-oss-high-vertex.json` records the Vertex-only routing
and shorter timeout used for the P3 high run; its model, reasoning level,
context limit and chunks otherwise match `gpt-oss-high.json`.
