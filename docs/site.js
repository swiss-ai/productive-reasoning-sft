(() => {
  'use strict';

  const data = window.SFT_REVIEW_DATA;
  const $ = id => document.getElementById(id);
  if (!data) {
    $('rollout').textContent = 'Rollout data could not be loaded.';
    return;
  }

  const annotated = new Map(data.cases.map(row => [row.id, row]));
  const records = [
    ...data.cases,
    ...data.audit.filter(row => !annotated.has(row.id)),
  ];
  const requested = decodeURIComponent(location.hash.slice(1));
  let selectedId = records.some(row => row.id === requested) ? requested : records[0].id;
  let visible = records;

  const add = (parent, tag, className = '', value = null) => {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (value !== null && value !== undefined) element.textContent = String(value);
    parent.appendChild(element);
    return element;
  };
  const compact = value => String(value ?? '').replace(/\s+/g, ' ').trim();
  const short = (value, length = 95) => {
    const text = compact(value);
    return text.length > length ? `${text.slice(0, length).trimEnd()}…` : text;
  };
  const sourceName = value => {
    const [source, task] = String(value ?? '').split(':');
    const name = source === 'reasoning_gym' ? 'Reasoning Gym' : source.split('/').at(-1);
    return task ? `${name} · ${task}` : name;
  };
  const display = value => value === null || value === undefined || value === '' ? 'Unavailable' : String(value);
  const friendly = value => display(value).replaceAll('_', ' ');
  const score = value => value === null || value === undefined ? 'unavailable' : `${value}/5`;

  function reviewRow(list, label, text, minor = null) {
    const row = add(list, 'div', 'review-row');
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
    add(head, 'h2', '', row.title || short(row.question, 125));
    add(head, 'div', 'record-id', `Candidate ${row.id}`);
    const meta = add(head, 'div', 'record-meta');
    add(meta, 'span', `decision ${row.selected ? 'pass' : 'reject'}`, row.selected ? 'Strict SFT: eligible' : 'Strict SFT: not eligible');
    add(meta, 'span', '', `Quality ${row.score}/5`);
    add(meta, 'span', '', `Correctness: ${friendly(row.correctness)}`);
    add(meta, 'span', '', `Hygiene: ${friendly(row.hygiene)}`);

    if (annotated.has(row.id)) {
      const note = add(article, 'section', 'annotation');
      add(note, 'h3', '', `Manual annotation · ${row.cohort}`);
      add(note, 'p', '', row.summary);
      add(note, 'p', '', row.why);
      if (row.quote) add(note, 'blockquote', '', row.quote);
    } else {
      add(article, 'p', 'annotation-absence', 'No manual annotation for this rollout; the review below is automated.');
    }

    const panes = add(article, 'div', 'panes');
    for (const [heading, value] of [
      ['Question', row.question],
      ['Reasoning trace', row.reasoning],
      ['Final answer', row.response],
    ]) {
      const pane = add(panes, 'section', 'pane');
      add(pane, 'h3', '', heading);
      add(pane, 'div', `pane-body${value ? '' : ' empty'}`, value || '(empty)');
    }

    const review = add(article, 'section', 'review');
    add(review, 'h3', '', 'Verification and judge review');
    add(review, 'p', 'review-note', 'The source check, model judgments, focused checks, and final selection are separate signals. Read disagreements rather than treating a score as ground truth.');
    const list = add(review, 'dl', 'review-list');
    reviewRow(
      list,
      'Source answer check',
      `${friendly(row.verification)}${row.verifier ? ` via ${friendly(row.verifier)}` : ''}`,
      'A source match checks the answer where possible; it does not certify the reasoning trace.'
    );
    reviewRow(
      list,
      'Correctness judge',
      row.judge?.correctness?.feedback || 'No parsed correctness feedback.',
      `Combined verdict: ${friendly(row.correctness)}${row.judge?.correctness?.verdict ? ` · judge: ${friendly(row.judge.correctness.verdict)}` : ''}`
    );
    reviewRow(
      list,
      'Reasoning judge',
      row.judge?.reasoning?.feedback || 'No parsed reasoning feedback.',
      `Reasoning score: ${score(row.judge?.reasoning?.score)}`
    );
    reviewRow(
      list,
      'Response judge',
      row.judge?.response?.feedback || 'No parsed response feedback.',
      `Response score: ${score(row.judge?.response?.score)}`
    );
    const hygiene = reviewRow(
      list,
      'Focused hygiene checks',
      `${friendly(row.hygiene)}${row.findings?.length ? ` · ${row.findings.length} non-clear finding${row.findings.length === 1 ? '' : 's'}` : ' · no non-clear findings recorded'}`
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
      `Finish reason: ${friendly(row.finish)} · reasoning ${row.reasoning_tokens ?? '—'} tokens · answer ${row.response_tokens ?? '—'} tokens`
    );
    reviewRow(
      list,
      'Strict SFT label',
      row.selected ? 'Eligible. This is a pipeline label, not independent human approval.' : 'Not eligible; the row is still retained.',
      row.exclusions?.length ? `Reasons: ${row.exclusions.map(friendly).join(', ')}` : `Quality score: ${row.score}/5`
    );
  }

  function optionLabel(row, index) {
    const note = annotated.has(row.id) ? 'Manual note · ' : '';
    const cohort = row.cohort.startsWith('Earlier') ? 'EARLIER' : 'NEW';
    return `${String(index + 1).padStart(2, '0')} · ${cohort} · ${note}${row.title || `${sourceName(row.source)}: ${short(row.question, 82)}`}`;
  }

  function updateChooser() {
    const query = compact($('rollout-search').value).toLowerCase();
    visible = records.filter(row => !query || `${row.id} ${row.source} ${row.question} ${row.title || ''} ${row.summary || ''}`.toLowerCase().includes(query));
    if (!visible.some(row => row.id === selectedId)) selectedId = visible[0]?.id;
    const select = $('rollout-select');
    select.replaceChildren();
    visible.forEach((row, index) => {
      const option = add(select, 'option', '', optionLabel(row, index));
      option.value = row.id;
    });
    select.value = selectedId || '';
    const index = visible.findIndex(row => row.id === selectedId);
    $('position').textContent = visible.length ? `${index + 1} of ${visible.length}` : '0 found';
    $('previous').disabled = index <= 0;
    $('next').disabled = index < 0 || index >= visible.length - 1;
    if (selectedId) history.replaceState(null, '', `#${selectedId}`);
    render();
  }

  function step(direction) {
    const index = visible.findIndex(row => row.id === selectedId);
    const next = visible[index + direction];
    if (next) {
      selectedId = next.id;
      updateChooser();
    }
  }

  $('batch-summary').textContent = `${data.batch.count} new rollouts · ${data.batch.passed} eligible · ${data.batch.rejected} not eligible · ${data.cases.length} annotated examples.`;
  $('rollout-search').addEventListener('input', updateChooser);
  $('rollout-select').addEventListener('change', event => { selectedId = event.target.value; updateChooser(); });
  $('previous').addEventListener('click', () => step(-1));
  $('next').addEventListener('click', () => step(1));
  updateChooser();
})();
