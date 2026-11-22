## What this release adds

<!-- Plain-language summary of the new or changed pages, and the one or two findings a
reader should take away. -->

## Changed pages

<!-- List each new or changed chapter, note and appendix. -->

## Quality control

- [ ] The Publish run on `drafts` passed: every test, and the render.
- [ ] The render has no warnings and no unresolved cross-references (`?@`).
- [ ] Every new number in the text is computed by `mibook` and pinned by a test.
- [ ] Every new proof has been reread end to end; any incomplete proof is titled
      "Proof sketch" and says what it assumes.
- [ ] Every new figure has been looked at, in the HTML and the PDF.
- [ ] New prose has had a humanizer pass and states which environmental quantity it is
      about.
- [ ] New citations are marked *full text* or *secondary* in the sources appendix.
- [ ] Status labels (exploratory, draft, stable, promoted) are correct.
- [ ] `CHANGELOG.md` and `ROADMAP.md` are updated.
- [ ] The reviewer has read the changed pages in the rendered site (the `github-pages`
      artifact of the drafts run, or a local `quarto preview`).

Merging this pull request publishes the book.
