# RRRIE Frontend Design Research

## Research Basis

This frontend should behave like a clinical command workspace, not a generic chatbot or decorative SaaS landing page. The most useful current references are:

- WCAG 2.2 and WAI notes: focus visibility, focus not obscured, target size, status semantics, contrast, and keyboard-safe controls.
  - https://www.w3.org/TR/WCAG22/
  - https://www.w3.org/WAI/standards-guidelines/wcag/new-in-22/
- Radix Primitives accessibility guidance: prefer native controls, correct roles, accessible labels, and expected keyboard behavior.
  - https://www.radix-ui.com/primitives/docs/overview/accessibility
- web.dev Core Web Vitals guidance: keep LCP, INP, and CLS measurable, avoid unnecessary JavaScript, and preserve layout stability.
  - https://web.dev/articles/vitals
- Chrome animation/compositing guidance: keep animation bounded, prefer transform/opacity, and avoid broad layer promotion unless it has a clear purpose.
  - https://developer.chrome.com/blog/re-rastering-composite
- Modern product-design references from premium SaaS, AI agent interfaces, 21st.dev-style component galleries, Vercel/Stripe/Linear/Raycast-level product surfaces, and clinical/technical dashboards. These are inspiration sources only; do not copy layouts directly.

## Product Direction

- Keep the clinical decision artifact central. Chat, trace, runtime, evidence, and controls are supporting surfaces.
- Prefer a quiet light interface: paper surfaces, precise borders, low shadow, cobalt progress, and limited semantic warning or danger.
- Use density intentionally. The analysis screen should feel fast and focused, while library/system views can be denser but must scan cleanly.
- Avoid generic AI design fingerprints: purple gradients, glowing cards, decorative blobs, fake bento dashboards, and excessive glass effects.

## Visual Rules

- Tokens are the source of truth. New colors, radii, spacing, and shadows should be semantic CSS variables before repeated use.
- Use neutral backgrounds for surfaces and reserve blue for primary actions, active state, progress, and informative status.
- Use warning and danger only for clinical/safety meaning.
- Keep card radius restrained: 8px to 14px for operational surfaces, larger only for the fixed composer shell.
- Borders should carry most depth. Shadows should be rare and used for overlays, drawers, composer, toasts, or active focus.
- Typography should be clinical and legible: compact headings, 13px to 14px body text, tabular numerals for metrics, no decorative display type.

## Layout Rules

- The analysis screen hierarchy is: run header, current progress, live activity, patient input, result artifact, composer.
- Details remain secondary by default. The drawer should be easy to open, keyboard safe, and not obscure the primary result.
- Mobile should not simply shrink desktop. Pipeline stages become horizontal, optional details move behind drawers, and the artifact remains reachable.
- Avoid stacking repeated cards inside larger cards. Use sections, separators, strips, lists, and tables for operational data.

## Component Rules

- Icon-only buttons must have accessible names.
- Controls should expose state through classes and ARIA, not inline styles or ad hoc utility strings.
- Use named classes for recurring UI: metric rows, section headers, progress bars, system stats, legends, and status dividers.
- Empty, loading, error, interrupted, and completed states must be real layouts with useful copy and recovery actions.

## Motion Rules

- Motion must explain state. Good uses: active stage pulse, progress fill, drawer enter/exit, toast entrance, result arrival.
- Avoid constant decorative motion. Idle state should be calm.
- Prefer opacity, transform, and bounded progress width transitions. Avoid layout-driving animation.
- `prefers-reduced-motion: reduce` must disable non-essential animation.

## Accessibility Rules

- Target WCAG 2.2 AA for focus visibility, keyboard operation, labels, contrast, and target size.
- Loading completion must be conveyed programmatically through status text or focus behavior.
- Do not place focusable controls inside `aria-hidden` regions.
- Progress indicators need labels and values when determinate.
- Keyboard users must be able to send, cancel, switch views, open/close details, dismiss toasts, and navigate result controls.

## Performance Rules

- Keep the static no-build architecture unless a clear need appears.
- Avoid new dependencies for polish. Native CSS and small vanilla JS are enough here.
- Reduce CSS duplication before adding new layers.
- Keep runtime interaction responsive; long-running clinical work should update status without blocking input or repaint.
- External fonts should be treated as optional enhancement; the fallback stack must still look professional.

## Current Project Priorities

1. Stabilize and simplify the existing CSS cascade.
2. Remove inline visual styling from `index.html` where it repeats.
3. Remove stale Tailwind-style utility toggles from `app.js`.
4. Make the analysis idle/running/completed/error states calmer and clearer.
5. Add regression checks for mojibake, stale utility strings, accessibility hooks, and static asset contracts.

## Applied In This Pass

- Added a v17 clinical command polish layer: quieter surface tokens, restrained borders, lower shadows, calmer active states, and stronger focus treatment.
- Converted key dynamic JS fragments from utility class strings to semantic project classes.
- Added accessible progressbar state text and status announcements for the analysis surface.
- Tightened mobile behavior so the analysis card, composer, and quick-start chips fit without document-level horizontal overflow.
