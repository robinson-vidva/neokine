# Security Policy

neokine is a **fully client-side** static web app (plus a local Streamlit app).
It has no backend, no accounts, and no server that stores user data — images,
video, and webcam frames are processed in the browser and never uploaded (see
the [Privacy Policy](PRIVACY.md)). This keeps the attack surface small.

## Reporting a vulnerability

If you find a security issue, please report it privately using GitHub's
**"Report a vulnerability"** feature under the repository's **Security** tab
(Security advisories), or open a minimal issue that does not disclose exploit
details. Please include:

- what the issue is and where (file / URL / component),
- steps to reproduce, and
- the potential impact.

We'll acknowledge and address valid reports as soon as we reasonably can. There
is no bug-bounty program.

## Scope

In scope: the web app in `docs/` and the local app in `app/`.
Out of scope: vulnerabilities in third-party dependencies (MediaPipe, the CDN,
GitHub Pages, Cloudflare) — please report those to the respective projects.
