# Model configurations

These are the five configurations needed to reproduce the reported study:

- `gpt-4.1.json`: GPT-4.1 generator;
- `gpt-oss-high.json`: GPT-OSS-120B with high reasoning effort;
- `gpt-oss-medium.json`: GPT-OSS-120B with medium reasoning effort;
- `gpt-5.6-luna.json`: GPT-5.6-Luna with reasoning disabled;
- `judge-gpt-4.1.json`: GPT-4.1 evaluation judge.

Credentials are read from the environment variables listed in
`../keys.env.example`. Newly generated files are written under `../runs/` and
are not part of the released paper results.
