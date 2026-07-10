# Privacy Policy

_Last updated: 2026-07-10_

**neokine** is a browser-based pose-visualization demo. It is designed to keep
your data on your own device. This policy explains, plainly, what does and does
not happen when you use it.

## The short version

- **Your images, videos, and webcam frames never leave your device.** All pose
  estimation runs locally in your browser.
- **Nothing you analyze is uploaded, stored on a server, or shared.**
- **No accounts, no cookies, no analytics, no tracking, no advertising.**

## What is processed, and where

- **Images / videos / webcam:** When you choose a file or start the webcam, the
  frames are decoded and analyzed entirely in your browser's memory. Processed
  frames are discarded when you close or reset the page. They are **not**
  uploaded and **not** written to any server or database.
- **Exports (PNG / CSV):** When you use "Save PNG" or "Save CSV", the file is
  generated in your browser and saved directly to your device by you. It is not
  transmitted anywhere.

## Local storage on your device

- neokine saves your **display preferences** (model size, thresholds, number of
  people, label mode, canvas interactions, and the out-of-frame toggle) in your
  browser's `localStorage`, under the key `neokine.settings`.
- This stays **on your device**, is never sent anywhere, and only makes the app
  reopen the way you left it. You can clear it any time by clearing your
  browser's site data. No cookies are used.

## Third parties that your browser contacts

To run, the app downloads a few assets at load time. These providers receive
only the **standard request metadata** any web request includes (such as your IP
address, the time, and which file was requested). They do **not** receive your
images, video, or webcam frames.

- **MediaPipe library + WASM runtime** — fetched from `cdn.jsdelivr.net` (jsDelivr CDN).
- **Pose model file** — fetched from `storage.googleapis.com` (Google Cloud Storage).
- **Hosting** — the site is served via **GitHub Pages**, and (on the
  `robinsonvidva.com` custom domain) fronted by **Cloudflare**. As hosting/CDN
  providers, they process standard server request logs.

If you run the app fully offline against locally hosted copies of these assets,
no third-party requests are made at all.

## Webcam

The live webcam mode requests camera access through your browser's standard
permission prompt. The feed is used only to draw the pose overlay in real time;
frames are processed in memory and discarded, never recorded and never uploaded.
You can revoke camera access at any time via your browser.

## Children's data

neokine is a general visualization demo, not a service that collects data. It is
**not** an infant monitoring or diagnostic tool and does not collect personal
information from anyone, including children.

## Changes

This policy may be updated as the project changes. The "last updated" date above
reflects the current version.

## Contact

Questions about privacy can be raised by opening an issue on the project's
GitHub repository.
