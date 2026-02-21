# Images and Math

Questions support markdown-style text and LaTeX-style math syntax in prompt and solution fields.

## Image handling

- Open question editor page.
- Copy an image to clipboard.
- Paste while editor page is focused (`Ctrl+V` / `Cmd+V`).
- App stores image directly in question YAML as:
  - `data_base64`
  - `mime_type`
  - `sha256`

No separate figure file is required.

## Portability

Because images are embedded, sending a single question YAML file is enough to share the full question content.

## Paste normalization

When possible, pasted scientific notation such as `7.30 x 10-26` is normalized to markdown math form:

- `7.30 \times 10^{-26}`
