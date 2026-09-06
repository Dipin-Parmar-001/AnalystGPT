# AnalystGPT — UI Design Specification (Current State)

> Baseline documentation of the existing React dashboard (`frontend/src`) as of v2.0.
> Purpose: (1) a single source of truth for the current design system, and (2) the input
> brief for a visual redesign pass (e.g. via Google Stitch) — "redesign *this*, not
> something new."

---

## 1. Product Overview

**AnalystGPT** is a multi-agent AI data-analytics workspace. A user uploads a dataset,
enters an analysis query, reviews a generated execution plan, watches specialized agents
(Planner → DataAnalyst → MLAgent → Reporter) execute live in a sandbox, and receives a
verified executive markdown report plus downloadable artifacts.

The product's identity keywords: **dark ops-console**, **mission control**, **live
telemetry**, **glassmorphism**, **monospace data**, **severity-colored verification**.

Tech stack: React 18 + Vite, Tailwind CSS 3 (custom token theme), lucide-react icons,
react-markdown + remark-gfm for report rendering. Single-page app, one screen with two
tabs, three modals, WebSocket-driven live updates.

---

## 2. Design Tokens

### 2.1 Color System (dark-only theme, `darkMode: 'class'` but always dark)

| Token | Value | Usage |
|---|---|---|
| `background` | `#0B0F19` | Page background (deep navy-black) |
| `surface-50` | `#1A2234` | Highest raised surface |
| `surface-100` | `#141B2D` | Card / panel background (primary surface) |
| `surface-200` | `#0F1523` | Input backgrounds, code blocks, console body |
| `surface-300` | `#0B0F19` | Deepest inset surface |
| `surface-border` | `#1E293B` | All card/border strokes (slate-800-ish) |
| `surface-hover` | `#243048` | Hover state for secondary buttons/rows |
| `brand-primary` | `#6366F1` (indigo-500) | Primary actions, active tab, focus rings, primary glow |
| `brand-primaryHover` | `#4F46E5` | Primary button hover |
| `brand-secondary` | `#38BDF8` (sky-400) | Accents, markdown H2, section icons, highlights |
| `brand-accent` | `#10B981` (emerald) | Success states |
| `brand-danger` | `#EF4444` | Destructive / terminate |
| `brand-warning` | `#F59E0B` | Warnings |
| `brand-purple` | `#A855F7` | Planning states |

**Semantic severity palette** (used for findings & statuses — always rendered as
`*-950/40–60%` background + `*-700/80%` border + `*-300/400` text + colored icon):

| Severity | Background | Border | Text/Icon |
|---|---|---|---|
| Success / verified | `emerald-950/40` | `emerald-700/60` | `emerald-300/400` |
| Warning / minor finding | `amber-950/40–50` | `amber-700/80` | `amber-300/400` |
| Critical / danger | `red-950/40–60` | `red-700/80` | `red-300/400` |
| Info / executing | `indigo-950/40–80` | `indigo-700/60` | `indigo-300/400` |
| Planning | `purple-950/80` | `purple-700/60` | `purple-300/400` |

**Agent identity colors** (consistent everywhere — navbar pills, plan badges, console
thought tags):

| Agent | Color family |
|---|---|
| PlannerAgent | Indigo (`indigo-950/40` bg, `indigo-300` text) |
| DataAnalystAgent | Emerald (`emerald-950/80` bg, `emerald-300` text) |
| MLAgent | Fuchsia (`fuchsia-950/80` bg, `fuchsia-300` text) |
| ReporterAgent | Amber (`amber-950/80` bg, `amber-300` text) |

**Text ramp:** headings `text-slate-100/200`, body `text-slate-300`, secondary
`text-slate-400/500`, disabled/tertiary `text-slate-500/600`. Page base text `#F1F5F9`.

### 2.2 Typography

| Role | Font | Notes |
|---|---|---|
| UI sans | Inter → system-ui fallback | Feature settings `cv02 cv03 cv04 cv11` enabled |
| Data / mono | JetBrains Mono → Fira Code → Consolas | All telemetry, labels, statuses, metrics |

Type patterns (informal scale):
- Page/brand title: `text-lg font-bold` (gradient-clipped white→slate-400)
- Section headers in cards: `text-xs font-bold uppercase tracking-wider font-mono`
- Body: `text-sm text-slate-300 leading-relaxed`
- Secondary/labels: `text-xs text-slate-400`
- Micro/meta: `text-[10px]`–`text-[11px] font-mono`
- Markdown report body: h1 `text-2xl bold`, h2 `text-xl semibold text-brand-secondary`,
  body `text-sm`, tables `text-xs`

### 2.3 Surfaces, Elevation & Effects

- **Glass panels** (`.glass-panel`): `rgba(20,27,45,0.75)` + `backdrop-blur(12px)` +
  `1px` border `rgba(30,41,59,0.8)` — used for navbar and major sections.
- **Glass cards** (`.glass-card`): lighter variant, `blur(8px)`.
- Standard card: `rounded-2xl bg-surface-100/90 border border-surface-border shadow-xl/lg`.
- **Glow shadows** (`.glow-*`): `0 0 15px` colored halo at 35% opacity — `glow-primary`
  (indigo), `glow-success` (emerald), `glow-danger` (red), `glow-warning` (amber).
- Animated glow: `glow` keyframes 10px→25px indigo (running states).
- Primary CTA: gradient `from-brand-primary to-indigo-600` with
  `shadow-lg shadow-brand-primary/20`.
- Danger CTA: gradient `from-red-600 to-rose-700`, `animate-pulse` while armed.
- Scrollbars: 6px, track `#0B0F19`, thumb `#1E293B` → `#334155` on hover.
- Motion: `animate-in fade-in` on mount, `animate-ping` dot on live indicators,
  `animate-pulse-subtle` (3s) on running badges. Durations ~150ms, ease transitions.

### 2.4 Shape & Spacing

- Radius: `rounded-2xl` (cards/panels/modals), `rounded-xl` (buttons, inner cards),
  `rounded-lg` (small chips), `rounded-full` (status pills, dots).
- Page gutter: `max-w-7xl mx-auto px-4/6/8 py-6`; card padding `p-4`/`p-5`;
  vertical rhythm `space-y-6` between major sections, `space-y-3/4` inside.
- Grid: main layout `grid-cols-12` → 4-col sidebar + 8-col workspace (`lg:` breakpoint);
  collapses to single column on mobile.

---

## 3. Layout Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Navbar (sticky, glass, h-16)                                     │
│  [Logo+brand+version pill] [4 agent pills] [status pill] [ws] [⟳]│
├──────────────────────────────────────────────────────────────────┤
│ StatusAlertBanner (conditional, full-width)                      │
├──────────────────────────────────────────────────────────────────┤
│ Query & Action Bar (glass panel)                                 │
│  [query input....................] [Plan Tasks] [Run Workflow]   │
│  Suggestions: [chip] [chip] [chip]                               │
├──────────────┬───────────────────────────────────────────────────┤
│ LEFT (4col)  │ RIGHT (8col)                                      │
│ Dataset      │ [Tab: Plan & Live Agent Stream | Report & Artifacts]│
│ Uploader/    │ ┌───────────────────────────────────────────────┐ │
│ Selector     │ │ Tab content                                   │ │
│ Quick Stats  │ │  Tab1: PlanViewer + AgentConsole              │ │
│ (plan id,    │ │  Tab2: FindingsPanel + ReportViewer +         │ │
│ tasks, arti- │ │        ArtifactsGallery                       │ │
│ facts, tool  │ └───────────────────────────────────────────────┘ │
│ budget)      │                                                   │
└──────────────┴───────────────────────────────────────────────────┘
Modals: TerminateModal, DatasetPreviewModal, Artifact lightbox
```

---

## 4. Component Inventory

### 4.1 Navbar
- Sticky, glass, 64px tall. Left: 40px gradient-rounded logo tile (Bot icon) + wordmark
  with gradient clip + `v2.0 Multi-Agent` mono badge + subtitle.
- Center (md+): four agent pills, each with 6px colored dot + agent name (mono, xs).
- Right: workflow status pill (state-driven, see §6), WebSocket health chip
  (`Live Sync` emerald / `Connecting...` amber, Wifi/WifiOff icons), Reset icon-button
  (disabled while running).

### 4.2 StatusAlertBanner (full-width, conditional)
| Status | Color scheme | Icon | Badge chip |
|---|---|---|---|
| `completed` | emerald | CheckCircle2 | `Verified` |
| `completed_with_warnings` | amber | ShieldAlert | `Review Findings` |
| `failed-consistency` | red | ShieldAlert | `Failed Consistency` |
| `cancelled` | red | Flame | `Terminated` (+ stopped-at step line) |
| `failed` | amber | AlertTriangle | `Diagnostic State` (+ error in mono code box) |

Layout: icon tile (32px, rounded-lg, darker bg + border) + bold title + explanation,
`animate-in fade-in`.

### 4.3 Query & Action Bar
- Glass panel. Uppercase mono label with BrainCircuit icon; targeting chip right-aligned.
- Large rounded input (surface-200, focus: indigo border + ring).
- Buttons: `Plan Tasks` (secondary, Sparkles icon) + `Run Workflow` (primary gradient,
  Play icon) → becomes red pulsing `Terminate Workflow` (Flame) while running.
- Suggestion chips row below (mono 11px, truncated, surface-200/60).

### 4.4 DatasetUploader (left column)
- Upload dropzone (CSV/XLSX), dataset list with radio-style multi-select checkboxes,
  filename + size + row/col meta, per-item preview button; "Targeting:" count chip.

### 4.5 Quick Stats (left column)
- Mono key-value rows on inset card: Active Plan ID, Completed Tasks x/y, Artifacts
  Produced, Max Tool Calls/Agent. Values `text-slate-200 font-semibold`.

### 4.6 PlanViewer (Tab 1)
- Header: ListChecks icon + "Execution Plan" + plan ID chip + agent count.
- Task list: rows with task number, title, agent badge (color-coded per §2.1),
  status badge (`Completed` emerald check / `In Progress` pulsing / `Pending` clock /
  `Failed` x-circle / `Cancelled`), description caption, expected-output mono line.
- Footer actions: `Confirm & Execute` (primary), `Modify Plan` expander with
  instruction input + Apply, `Cancel Plan` (secondary).

### 4.7 AgentConsole (Tab 1)
- 520px fixed-height terminal panel, header with Terminal icon + `Agent Live Stream` +
  pulsing "Streaming" dot while running + agent filter dropdown + Auto-Scroll toggle.
- Event card types (chronological, mono, auto-scroll):
  - **LLM Thought**: agent-colored tag + full thought prose (surface-100 card).
  - **Tool Executing**: indigo-tinted card, chevron-expand, Python code in dark `<pre>`
    with emerald text, tool name + call counter.
  - **Tool Completed**: stdout (mono pre), stderr (red-tinted pre), saved-artifacts line.
  - **Task Start / Completed**: slim full-width pills (indigo / emerald).
  - **Final Verification Started**: indigo pill "numeric gate + consistency checker +
    LLM judge" (ShieldCheck).
  - **Report Repair Started**: amber card "Repair Pass n/2 — N critical finding(s)"
    (ShieldAlert) explaining combined correction prompt + 5-call budget.
  - **Report Repair Completed**: emerald pill "re-verifying with all methods".
  - **Workflow Cancelled / Failed**: full-width red/amber banners with message.
  - Generic fallback row for any other event.

### 4.8 FindingsPanel (Tab 2, above report)
- Header: ShieldAlert icon + `VERIFICATION FINDINGS` + severity count chips
  (red "N critical", amber "N minor") + optional "N repair pass(es)" neutral chip.
- Status banners inside: `failed-consistency` (red: "Run NOT cleared for delivery —
  treat every figure as unverified"), `completed_with_warnings` (amber), clean
  `completed` renders a compact emerald "Verification Passed" card instead.
- Finding cards: severity icon + `[SEVERITY]` mono tag + type label
  (`Shape mismatch`, `Phantom columns`, `Stale champion pointer`,
  `Contradicted claim (judge)`, `Unverified numeric claim`, …) + truncated message.
  Expandable: full message, grounding line, artifact list, source
  (`numeric_verifier` / `consistency_judge`). Critical cards default open.
- Known finding-type labels are mapped to human labels; unknown codes render raw.

### 4.9 ReportViewer (Tab 2)
- Header: FileText icon + `EXECUTIVE ANALYSIS REPORT` + Raw Markdown toggle, Copy
  (2s check feedback), Download .md buttons.
- Body: 600px max-height scroll, rendered markdown (`.markdown-body` styles: styled
  h1/h2 with borders, tables with surface headers, indigo blockquotes, mono code) or
  raw pre view.
- Empty state: centered FileText icon + "No Report Generated Yet".

### 4.10 ArtifactsGallery (Tab 2)
- Responsive 3-col grid (1/2/3 by breakpoint) of artifact cards: type icon
  (image/dataset/json/report/model), filename, size, modified date; image artifacts
  show thumbnail; click → fullscreen lightbox (`bg-black/85 blur`) with max-w-4xl
  framed preview; download affordance per artifact.

### 4.11 Modals
- **TerminateModal**: two-step destructive confirm. Overlay `black/80 + blur`; card
  `max-w-md rounded-2xl border-red-800/80 glow-danger`; red confirm button; used
  mid-run (also reachable from tab-bar "Stop Execution").
- **DatasetPreviewModal**: `max-w-4xl`, schema/preview table of uploaded CSV.
- **Artifact lightbox**: as above.

### 4.12 Iconography
- lucide-react exclusively, stroke style, 14–24px, always color-matched to context
  (BrainCircuit, Bot, BarChart3, Cpu, FileText, Terminal, ShieldCheck, ShieldAlert,
  Flame, Play, Sparkles, RefreshCw, Wifi/WifiOff, AlertTriangle, AlertOctagon…).

---

## 5. Workflow Lifecycle (state → UI mapping)

| State | Navbar pill | Banner | Notes |
|---|---|---|---|
| `idle` | grey "System Idle" | — | — |
| `planning` | purple "Planning Tasks" | — | — |
| `plan_ready` | sky "Plan Ready" | — | PlanViewer active |
| `executing` | indigo pulsing "Executing Pipeline" | — | Terminate CTAs armed, console streaming |
| `completed` | emerald "Workflow Completed" | green "Verified" | FindingsPanel shows pass/fail detail |
| `completed_with_warnings` | (banner-driven) | amber | findings panel expanded |
| `failed-consistency` | (banner-driven) | red | report visually demoted, findings critical-first |
| `cancelled` | red "Workflow Terminated" | red + stopped-at step | — |
| `failed` | amber "Execution Error" | amber + error box | — |

Verification data flow: WebSocket `workflow_completed` carries `status`,
`severity_counts`, `findings`, `repair_attempts_used`; REST fallbacks
`/api/workflow/status`, `/api/workflow/findings`, `/api/report`.

---

## 6. Voice & Tone

- Labels: uppercase, mono, tracked-wide (`EXECUTIVE ANALYSIS REPORT`).
- Status language: verification-centric — "cleared for delivery", "unverified",
  "contradicted claim", "repair pass", "not cleared".
- Numbers/artifacts always mono; never rounded corners on data (16px cards, 8px chips).

---

## 7. Notes for Redesign (Stitch Brief)

**Keep (identity):** dark navy palette + indigo/sky brand, glassmorphism, mono
telemetry, agent color-coding, severity color language, two-tab structure,
live-stream console concept.

**Improve candidates:**
1. FindingsPanel could graduate from accordion cards to a triage table/timeline view
   with filters (by severity/source/artifact) and a severity sparkline.
2. AgentConsole could adopt a proper terminal aesthetic (toolbar tabs per agent,
   event-type icons, color-coded left border per severity) and collapsible run phases
   (plan → tasks → verification → repair).
3. Report tab could get a document-style layout (fixed TOC rail, section anchors,
   reading width) instead of a single scroll pane.
4. Status/verification could be a persistent right rail (always-visible run health)
   rather than top-of-tab panels.
5. Quick Stats could become a sparkline/metric-tile row with task progress ring.

Constraints: React 18 + Tailwind (tokens in §2 may be extended, not renamed),
lucide-react icons only, must keep WebSocket event contract and all component props,
must preserve severity color semantics (§2.1) and agent color-coding.
