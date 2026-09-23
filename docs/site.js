(() => {
  'use strict';

  const data = window.SFT_REVIEW_DATA;
  if (!data) {
    document.getElementById('case-detail').textContent = 'Review data could not be loaded.';
    return;
  }

  const $ = id => document.getElementById(id);
  const state = {caseId: data.cases[0]?.id, caseFilter: 'all', auditId: data.audit[0]?.id};
  const add = (parent, tag, className = '', value = null) => {
    const item = document.createElement(tag);
    if (className) item.className = className;
    if (value !== null && value !== undefined) item.textContent = String(value);
    parent.appendChild(item);
    return item;
  };
  const compact = value => String(value ?? '').replace(/\s+/g, ' ').trim();
  const short = (value, max = 160) => {
    const content = compact(value);
    return content.length > max ? `${content.slice(0, max).trimEnd()}…` : content;
  };
  const sourceName = source => {
    const [dataset, task] = String(source ?? '').split(':');
    const name = dataset === 'reasoning_gym' ? 'Reasoning Gym' : dataset.split('/').at(-1);
    return task ? `${name} · ${task}` : name;
  };
  const pill = (parent, value, style = 'neutral') => add(parent, 'span', `pill ${style}`, value);
  const status = row => row.selected ? 'Passed' : 'Not eligible';

  function addRaw(parent, row) {
    const wrapper = add(parent, 'div', 'raw-disclosure');
    add(wrapper, 'h4', '', 'Original record');
    for (const [label, value] of [
      ['Original question', row.question],
      ['Full reasoning trace', row.reasoning],
      ['Final response', row.response],
    ]) {
      const details = add(wrapper, 'details');
      add(details, 'summary', '', label);
      add(details, 'pre', '', value || '(empty)');
    }
  }

  function addEvidence(parent, row) {
    add(parent, 'h4', '', 'Pipeline evidence');
    const grid = add(parent, 'div', 'evidence-grid');
    const values = [
      ['Source check', `${row.verification || 'unavailable'} · ${row.verifier || 'none'}`],
      ['Answer verdict', row.correctness || 'unknown'],
      ['Hygiene', row.hygiene || 'unknown'],
      ['Reasoning / answer tokens', `${row.reasoning_tokens ?? '—'} / ${row.response_tokens ?? '—'}`],
    ];
    for (const [label, value] of values) {
      const card = add(grid, 'div');
      add(card, 'small', '', label);
      add(card, 'strong', '', value);
    }
    if (row.exclusions?.length) {
      add(parent, 'h4', '', 'Exclusion reasons');
      add(parent, 'p', 'source-line', row.exclusions.join(' · '));
    }
  }

  function addFindings(parent, row) {
    if (!row.findings?.length) return;
    add(parent, 'h4', '', 'Focused hygiene findings');
    for (const finding of row.findings) {
      const box = add(parent, 'div', 'finding');
      add(box, 'strong', '', `${String(finding.category || '').replaceAll('_', ' ')} · ${finding.verdict}`);
      add(box, 'p', '', finding.explanation || 'No explanation recorded.');
      for (const evidence of Array.isArray(finding.evidence) ? finding.evidence : []) {
        add(box, 'blockquote', '', evidence);
      }
    }
  }

  function addJudge(parent, row) {
    add(parent, 'h4', '', 'Judge review');
    const grid = add(parent, 'div', 'judge-grid');
    for (const label of ['correctness', 'reasoning', 'response']) {
      const part = row.judge?.[label] || {};
      const card = add(grid, 'div', 'judge-card');
      add(card, 'strong', '', `${label}${part.score === null || part.score === undefined ? '' : ` · ${part.score}/5`}`);
      add(card, 'p', '', part.feedback || part.verdict || 'No parsed review.');
    }
  }

  function renderCaseList() {
    const list = $('case-list');
    list.replaceChildren();
    const visible = data.cases.filter(row =>
      state.caseFilter === 'all' ||
      (state.caseFilter === 'pass' && row.selected) ||
      (state.caseFilter === 'reject' && !row.selected) ||
      (state.caseFilter === 'earlier' && row.cohort.startsWith('Earlier'))
    );
    if (!visible.some(row => row.id === state.caseId)) state.caseId = visible[0]?.id;
    visible.forEach((row, index) => {
      const card = add(list, 'button', `case-card ${row.selected ? 'pass' : 'reject'}${row.id === state.caseId ? ' active' : ''}`);
      card.type = 'button';
      card.setAttribute('aria-pressed', String(row.id === state.caseId));
      const top = add(card, 'span', 'card-top');
      add(top, 'span', 'card-index', String(index + 1).padStart(2, '0'));
      pill(top, status(row), row.selected ? 'pass' : 'reject');
      add(card, 'span', 'card-title', row.title);
      add(card, 'span', 'card-meta', row.cohort);
      card.addEventListener('click', () => {
        state.caseId = row.id;
        renderCaseList();
        renderCaseDetail();
        if (window.innerWidth < 700) $('case-detail').scrollIntoView({behavior: 'smooth', block: 'start'});
      });
    });
    renderCaseDetail();
  }

  function renderCaseDetail() {
    const parent = $('case-detail');
    parent.replaceChildren();
    const row = data.cases.find(item => item.id === state.caseId);
    if (!row) { add(parent, 'p', 'empty', 'Choose an example.'); return; }
    const top = add(parent, 'div', 'detail-kicker');
    pill(top, row.cohort);
    pill(top, status(row), row.selected ? 'pass' : 'reject');
    pill(top, `Quality ${row.score}/5`);
    add(parent, 'h3', '', row.title);
    add(parent, 'p', 'detail-lead', row.summary);
    add(parent, 'p', 'source-line', `${sourceName(row.source)} · candidate ${row.id.slice(0, 12)}`);
    const verdict = add(parent, 'div', `verdict-box${row.selected ? '' : ' reject'}`);
    add(verdict, 'span', '', 'Why this example matters');
    add(verdict, 'p', '', row.why);
    if (row.quote) add(parent, 'blockquote', 'quote', row.quote);
    addEvidence(parent, row);
    addFindings(parent, row);
    addJudge(parent, row);
    addRaw(parent, row);
  }

  function auditRows() {
    const query = compact($('audit-search').value).toLowerCase();
    const filter = $('audit-filter').value;
    return data.audit.filter(row => {
      if (filter === 'pass' && !row.selected) return false;
      if (filter === 'reject' && row.selected) return false;
      return !query || `${row.question} ${row.source} ${row.id}`.toLowerCase().includes(query);
    });
  }

  function renderAuditList() {
    const visible = auditRows();
    const list = $('audit-list');
    list.replaceChildren();
    $('audit-count').textContent = `${visible.length} of ${data.audit.length} shown`;
    if (!visible.some(row => row.id === state.auditId)) state.auditId = visible[0]?.id;
    if (!visible.length) add(list, 'p', 'empty', 'No rollouts match this search.');
    visible.forEach(row => {
      const card = add(list, 'button', `case-card audit-card ${row.selected ? 'pass' : 'reject'}${row.id === state.auditId ? ' active' : ''}`);
      card.type = 'button';
      card.setAttribute('aria-pressed', String(row.id === state.auditId));
      const top = add(card, 'span', 'card-top');
      add(top, 'span', 'card-index', sourceName(row.source));
      pill(top, `Score ${row.score}`, row.selected ? 'pass' : 'reject');
      add(card, 'span', 'card-title', short(row.question, 170));
      add(card, 'span', 'card-meta', `${status(row)} · ${row.id.slice(0, 12)}`);
      card.addEventListener('click', () => {
        state.auditId = row.id;
        renderAuditList();
        renderAuditDetail();
        if (window.innerWidth < 700) $('audit-detail').scrollIntoView({behavior: 'smooth', block: 'start'});
      });
    });
    renderAuditDetail();
  }

  function renderAuditDetail() {
    const parent = $('audit-detail');
    parent.replaceChildren();
    const row = data.audit.find(item => item.id === state.auditId);
    if (!row) { add(parent, 'p', 'empty', 'Choose a rollout.'); return; }
    const top = add(parent, 'div', 'detail-kicker');
    pill(top, 'New 50-rollout batch');
    pill(top, status(row), row.selected ? 'pass' : 'reject');
    pill(top, `Quality ${row.score}/5`);
    add(parent, 'h3', '', 'Rollout review');
    add(parent, 'p', 'detail-lead', short(row.question, 260));
    add(parent, 'p', 'source-line', `${sourceName(row.source)} · candidate ${row.id}`);
    const verdict = add(parent, 'div', `verdict-box${row.selected ? '' : ' reject'}`);
    add(verdict, 'span', '', 'Pipeline decision');
    add(verdict, 'p', '', row.selected
      ? 'This row passes the current strict SFT view. That is not independent human certification.'
      : `This row remains retained but is not in the strict SFT view${row.exclusions?.length ? `: ${row.exclusions.join(', ')}` : '.'}`);
    addEvidence(parent, row);
    addFindings(parent, row);
    addJudge(parent, row);
    addRaw(parent, row);
  }

  $('batch-total').textContent = data.batch.count;
  $('batch-pass').textContent = data.batch.passed;
  $('batch-reject').textContent = data.batch.rejected;
  $('case-filters').querySelectorAll('button').forEach(button => button.addEventListener('click', () => {
    state.caseFilter = button.dataset.filter;
    $('case-filters').querySelectorAll('button').forEach(item => item.classList.toggle('active', item === button));
    renderCaseList();
  }));
  $('audit-search').addEventListener('input', renderAuditList);
  $('audit-filter').addEventListener('change', renderAuditList);
  renderCaseList();
  renderAuditList();
})();
