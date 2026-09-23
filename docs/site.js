(() => {
  'use strict';

  const data = window.SFT_REVIEW_DATA;
  const $ = id => document.getElementById(id);
  if (!data) {
    $('rollout').textContent = 'Rollout data could not be loaded.';
    return;
  }

  const records = [
    ...data.audit,
    ...data.cases.filter(row => row.cohort.startsWith('Earlier')),
  ];
  const requested = decodeURIComponent(location.hash.slice(1));
  let selectedId = records.some(row => row.id === requested) ? requested : records[0].id;
  const indexButtons = new Map();
  const markdown = window.markdownit && window.texmath && window.katex
    ? window.markdownit({ html: false, linkify: false }).use(window.texmath, {
      engine: window.katex,
      delimiters: ['dollars', 'brackets', 'beg_end'],
      katexOptions: { throwOnError: false, trust: false, maxExpand: 1000, maxSize: 10 },
    })
    : null;

  const add = (parent, tag, className = '', value = null) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (value !== null && value !== undefined) element.textContent = String(value);
    parent.appendChild(element);
    return element;
  };
  const sourceName = value => {
    const [source, task] = String(value ?? '').split(':');
    const name = source === 'reasoning_gym' ? 'Reasoning Gym' : source.split('/').at(-1);
    return task ? `${name} · ${task}` : name;
  };
  const display = value => value === null || value === undefined || value === '' ? 'Unavailable' : String(value);
  const friendly = value => display(value).replaceAll('_', ' ');
  const score = value => value === null || value === undefined ? 'unavailable' : `${value}/5`;

  function renderMarkdown(parent, value) {
    if (!value) {
      parent.classList.add('empty');
      parent.textContent = '(empty)';
      return;
    }
    if (markdown) {
      try {
        // Markdown treats a multiline \[...\] block as escaped brackets;
        // convert only standalone display delimiters before parsing.
        const mathReady = value
          .replace(/^[ \t]*\\\[[ \t]*$/gm, () => '\n$$')
          .replace(/^[ \t]*\\\][ \t]*$/gm, () => '$$\n');
        parent.innerHTML = markdown.render(mathReady);
        return;
      } catch (error) {
        console.warn('Markdown rendering failed; showing original text.', error);
      }
    }
    parent.classList.add('raw-fallback');
    parent.textContent = value;
  }

  function reviewRow(list, label, text, minor = null, tone = '') {
    const row = add(list, 'div', `review-row${tone ? ` tone-${tone}` : ''}`);
    add(row, 'dt', '', label);
    const detail = add(row, 'dd', '', text);
    if (minor) add(detail, 'span', 'minor', minor);
    return detail;
  }

  function render() {
    const row = records.find(item => item.id === selectedId);
    const article = $('rollout');
    article.replaceChildren();
    if (!row) return;

    const head = add(article, 'header', 'record-head');
    const top = add(head, 'div', 'record-top');
    add(top, 'span', row.cohort.startsWith('Earlier') ? 'cohort-earlier' : '', row.cohort);
    add(top, 'span', '', '·');
    add(top, 'span', '', sourceName(row.source));
    add(head, 'h2', '', `Rollout ${String(records.indexOf(row) + 1).padStart(2, '0')}`);
    add(head, 'div', 'record-id', `Candidate ${row.id}`);
    const meta = add(head, 'div', 'record-meta');
    add(meta, 'span', `decision ${row.selected ? 'pass' : 'reject'}`, row.selected ? 'Strict SFT: eligible' : 'Strict SFT: not eligible');
    add(meta, 'span', '', `Quality ${row.score}/5`);
    add(meta, 'span', '', `Correctness: ${friendly(row.correctness)}`);
    add(meta, 'span', '', `Hygiene: ${friendly(row.hygiene)}`);

    const panes = add(article, 'div', 'panes');
    for (const [heading, value] of [
      ['Question', row.question],
      ['Reasoning trace', row.reasoning],
      ['Final answer', row.response],
    ]) {
      const pane = add(panes, 'section', 'pane');
      add(pane, 'h3', '', heading);
      renderMarkdown(add(pane, 'div', 'pane-body'), value);
    }

    const review = add(article, 'section', 'review');
    add(review, 'h3', '', 'Verification and judge review');
    add(review, 'p', 'review-note', 'The source check, model judgments, focused checks, and final selection are separate signals. Read disagreements rather than treating a score as ground truth.');
    const list = add(review, 'dl', 'review-list');
    reviewRow(
      list,
      'Source answer check',
      `${friendly(row.verification)}${row.verifier ? ` via ${friendly(row.verifier)}` : ''}`,
      'A source match checks the answer where possible; it does not certify the reasoning trace.',
      row.verification === 'passed' ? 'pass' : row.verification === 'failed' ? 'reject' : 'warn'
    );
    reviewRow(
      list,
      'Correctness judge',
      row.judge?.correctness?.feedback || 'No parsed correctness feedback.',
      `Combined verdict: ${friendly(row.correctness)}${row.judge?.correctness?.verdict ? ` · judge: ${friendly(row.judge.correctness.verdict)}` : ''}`,
      row.correctness === 'incorrect' ? 'reject' : ['verified', 'supported', 'correct'].includes(row.correctness) ? 'pass' : 'warn'
    );
    reviewRow(
      list,
      'Reasoning judge',
      row.judge?.reasoning?.feedback || 'No parsed reasoning feedback.',
      `Reasoning score: ${score(row.judge?.reasoning?.score)}`,
      row.judge?.reasoning?.score >= 4 ? 'pass' : row.judge?.reasoning?.score <= 2 ? 'reject' : 'warn'
    );
    reviewRow(
      list,
      'Response judge',
      row.judge?.response?.feedback || 'No parsed response feedback.',
      `Response score: ${score(row.judge?.response?.score)}`,
      row.judge?.response?.score >= 4 ? 'pass' : row.judge?.response?.score <= 2 ? 'reject' : 'warn'
    );
    const hygiene = reviewRow(
      list,
      'Focused hygiene checks',
      `${friendly(row.hygiene)}${row.findings?.length ? ` · ${row.findings.length} non-clear finding${row.findings.length === 1 ? '' : 's'}` : ' · no non-clear findings recorded'}`,
      null,
      row.hygiene === 'passed' ? 'pass' : row.hygiene === 'failed' ? 'reject' : 'warn'
    );
    for (const finding of row.findings || []) {
      const item = add(hygiene, 'div', 'minor');
      add(item, 'span', '', `${friendly(finding.category)} — ${friendly(finding.verdict)}: ${finding.explanation || 'No explanation recorded.'}`);
      for (const evidence of Array.isArray(finding.evidence) ? finding.evidence : []) {
        add(item, 'blockquote', '', evidence);
      }
    }
    reviewRow(
      list,
      'Completion and length',
      `Finish reason: ${friendly(row.finish)} · reasoning ${row.reasoning_tokens ?? '—'} tokens · answer ${row.response_tokens ?? '—'} tokens`,
      null,
      row.finish === 'length' ? 'reject' : row.finish === 'stop' ? 'pass' : 'warn'
    );
    reviewRow(
      list,
      'Strict SFT label',
      row.selected ? 'Eligible. This is a pipeline label, not independent human approval.' : 'Not eligible; the row is still retained.',
      row.exclusions?.length ? `Reasons: ${row.exclusions.map(friendly).join(', ')}` : `Quality score: ${row.score}/5`,
      row.selected ? 'pass' : 'reject'
    );
  }

  function selectRollout(id) {
    selectedId = id;
    for (const [rowId, button] of indexButtons) {
      const active = rowId === id;
      button.classList.toggle('active', active);
      if (active) button.setAttribute('aria-current', 'true');
      else button.removeAttribute('aria-current');
    }
    history.replaceState(null, '', `#${encodeURIComponent(id)}`);
    render();
    window.scrollTo({ top: 0 });
  }

  $('batch-summary').textContent = `${data.batch.count} new rollouts · ${data.batch.passed} eligible · ${data.batch.rejected} not eligible.`;
  $('rollout-count').textContent = records.length;
  for (const [index, row] of records.entries()) {
    const button = add($('rollout-index'), 'button', `index-item ${row.selected ? 'pass' : 'reject'}`);
    button.type = 'button';
    button.setAttribute('aria-label', `Rollout ${index + 1}: ${row.selected ? 'eligible' : 'not eligible'}, quality ${row.score} of 5`);
    add(button, 'span', 'index-number', String(index + 1).padStart(2, '0'));
    const meta = add(button, 'span', 'index-meta');
    add(meta, 'span', 'index-decision', row.selected ? 'eligible' : 'not eligible');
    add(meta, 'span', 'index-score', `${row.score}/5`);
    button.addEventListener('click', () => selectRollout(row.id));
    indexButtons.set(row.id, button);
  }
  selectRollout(selectedId);
})();
