// Load insights and actions from results/insights.txt via fetch
async function loadInsights() {
  try {
    const resp = await fetch('../results/insights.txt');
    if (!resp.ok) throw new Error('insights file not found');
    const text = await resp.text();
    parseInsights(text);
  } catch (e) {
    console.warn('Could not load insights:', e);
    document.getElementById('insights-cards').innerHTML = '<div class="card"><p class="muted">Insights file not available.</p></div>';
    document.getElementById('actions').innerHTML = '<div class="card"><p class="muted">Action plan not available.</p></div>';
  }
}

function parseInsights(text) {
  const sections = text.split(/\n\n/).map(s => s.trim()).filter(Boolean);
  // crude but effective: find the KEY INSIGHTS block and PRACTICAL ACTION PLAN
  const insightsStart = text.indexOf('KEY INSIGHTS');
  const actionStart = text.indexOf('PRACTICAL ACTION PLAN');
  const insightsText = insightsStart >= 0 && actionStart > insightsStart ? text.slice(insightsStart, actionStart) : '';
  const actionsText = actionStart >= 0 ? text.slice(actionStart) : '';

  const insightsContainer = document.getElementById('insights-cards');
  insightsContainer.innerHTML = '';

  // Extract the 7 insights by splitting on 'Insight'
  const matches = text.split(/Insight \d+: /).slice(1);
  matches.forEach(m => {
    const lines = m.split('\n').map(l => l.trim()).filter(Boolean);
    const title = lines[0] || '';
    const finding = lines.slice(1,4).join('\n');
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `<h4>${title}</h4><pre style="white-space:pre-wrap">${finding}</pre>`;
    insightsContainer.appendChild(card);
  });

  // Actions
  const actionsContainer = document.getElementById('actions');
  actionsContainer.innerHTML = '';
  const actionBlocks = actionsText.split(/\n\n/).filter(s=>/Action/gi.test(s));
  if (actionBlocks.length===0){
    actionsContainer.innerHTML = '<div class="card"><p class="muted">No actions parsed.</p></div>';
    return;
  }
  actionBlocks.forEach(b=>{
    const level = b.match(/\[([A-Z]+)\]/); // [HIGH]
    const badge = level ? level[1] : 'MEDIUM';
    const titleLine = b.split('\n').find(l=>l.startsWith('Action')) || '';
    const detail = b.split('\n').slice(1).join('\n');
    const card = document.createElement('div');
    card.className = 'card';
    card.innerHTML = `<div style="display:flex;justify-content:space-between;align-items:center"><h4>${titleLine}</h4><div class="badge ${badge.toLowerCase()}">${badge}</div></div><pre style="white-space:pre-wrap">${detail}</pre>`;
    actionsContainer.appendChild(card);
  });
}

document.addEventListener('DOMContentLoaded', ()=>{
  loadInsights();
});
