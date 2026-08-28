# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for the BraTS orchestrator.

## What is an ADR?

An ADR captures a significant architectural decision along with its context, the options considered, and the consequences. It serves as a historical record for current and future maintainers to understand *why* the system is built the way it is.

## When to write an ADR

Write an ADR when making a decision that:

- Affects the package's architecture, structure, or dependencies
- Introduces a new pattern or deprecates an existing one
- Has non-obvious trade-offs that future contributors should understand
- Changes the public API or supported backends

Not every decision needs an ADR — routine refactoring, bug fixes, and cosmetic changes do not.

## How to create a new ADR

1. Copy `0000-template.md` to a new file numbered sequentially (e.g., `0003-my-decision.md`)
2. Fill in each section
3. Set the status to `Proposed`
4. Open a pull request for discussion
5. Once accepted, update the status to `Accepted`

## Status conventions

| Status       | Meaning                                          |
|--------------|--------------------------------------------------|
| Proposed     | Under discussion, not yet adopted                |
| Accepted     | Approved and implemented                         |
| Deprecated   | Still in effect but planned for removal          |
| Superseded   | Replaced by a later ADR (reference the new ADR)  |

## Index

| Number | Title                                      | Status   |
|--------|--------------------------------------------|----------|
| 0001   | Container orchestration via Template Method and Strategy | Accepted |
| 0002   | YAML-driven algorithm registry             | Accepted |
