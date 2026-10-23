// Run with: node --test sam2_pipeline/scripts/test_canvas_active_label.cjs
// Synthetic DOM state only; no Gradio server or participant files are needed.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const source = fs.readFileSync(process.env.CANVAS_SOURCE || path.join(__dirname, 'prompter_canvas_fast.py'), 'utf8');
const match = source.match(/  function activeLabel\(\) \{[\s\S]*?\n  \}/);
assert.ok(match, 'Find the actual canvas label-selection function');

function harness(checkedLabel, bridgeLabel, cachedLabel = '') {
  const selection = { checkedLabel, bridgeLabel };
  const context = {
    state: { selectedLabel: cachedLabel },
    document: { querySelector: () => selection.checkedLabel === null ? null : { value: selection.checkedLabel } },
    componentInput: () => selection.bridgeLabel === null ? null : { value: selection.bridgeLabel },
  };
  vm.createContext(context);
  vm.runInContext(match[0], context);
  return { context, selection, read: () => context.activeLabel() };
}

test('selected radio wins while the bridge still has the previous label', () => {
  const h = harness('Distractor', 'Target', 'Target');
  assert.equal(h.read(), 'Distractor');
  assert.equal(h.context.state.selectedLabel, 'Distractor');
});

test('rapid switches follow the current radio in both directions', () => {
  const h = harness('Distractor', 'Target');
  assert.equal(h.read(), 'Distractor');
  h.selection.checkedLabel = 'Target';
  h.selection.bridgeLabel = 'Distractor';
  assert.equal(h.read(), 'Target');
});

test('bridge remains the fallback before a radio exists', () => {
  assert.equal(harness(null, 'Target').read(), 'Target');
});

test('cached label or empty string survives missing controls', () => {
  assert.equal(harness(null, null, 'Target').read(), 'Target');
  assert.equal(harness(null, null).read(), '');
});
