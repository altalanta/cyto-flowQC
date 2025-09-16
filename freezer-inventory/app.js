// UI logic and event handlers
(function(){
  const $ = sel => document.querySelector(sel);
  const $$ = sel => Array.from(document.querySelectorAll(sel));
  const toastEl = $('#toast');
  function toast(msg, kind='info', timeout=2500){
    toastEl.textContent = msg;
    toastEl.classList.add('show');
    setTimeout(()=>toastEl.classList.remove('show'), timeout);
  }

  // PWA registration (only under http/https)
  if (location.protocol.startsWith('http') && 'serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('./service-worker.js').catch(()=>{});
    });
  }

  // Elements
  const searchInput = $('#search');
  const locFilter = $('#locationFilter');
  const openedOnly = $('#openedOnly');
  const expFilter = $('#expFilter');
  const listEl = $('#list');
  const emptyEl = $('#emptyState');
  const loadSamplesBtn = $('#loadSamples');
  const addForm = $('#addForm');
  const addValidation = $('#addValidation');
  const foodInput = $('#foodInput');
  const typeahead = $('#foodTypeahead');
  const scrollToAdd = $('#scrollToAdd');
  const exportBtn = $('#exportBtn');
  const importFile = $('#importFile');

  let foodsCache = [];

  // Typeahead logic
  function renderTypeahead() {
    const q = foodInput.value.trim().toLowerCase();
    const matches = foodsCache
      .filter(f => !q || f.name.toLowerCase().includes(q) || (Array.isArray(f.aliases) && f.aliases.join(' ').toLowerCase().includes(q)))
      .slice(0,6);
    const options = [];
    matches.forEach((f,i)=>{
      options.push(`<div role="option" data-name="${encodeURIComponent(f.name)}" class="${i===0?'active':''}">${escapeHtml(f.name)}</div>`);
    });
    if (!matches.find(f => f.name.toLowerCase() === q) && q) {
      options.push(`<div role="option" data-create="${encodeURIComponent(foodInput.value.trim())}">Create “${escapeHtml(foodInput.value.trim())}”</div>`);
    }
    typeahead.innerHTML = options.join('');
  }

  function escapeHtml(s){return s.replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));}

  typeahead.addEventListener('click', (e)=>{
    const opt = e.target.closest('[role=option]');
    if (!opt) return;
    const create = opt.getAttribute('data-create');
    if (create) {
      foodInput.value = decodeURIComponent(create);
    } else {
      foodInput.value = decodeURIComponent(opt.getAttribute('data-name'));
    }
    typeahead.innerHTML = '';
    $('#qtyValue').focus();
  });

  foodInput.addEventListener('input', renderTypeahead);
  foodInput.addEventListener('focus', renderTypeahead);
  foodInput.addEventListener('blur', ()=> setTimeout(()=> typeahead.innerHTML = '', 120));
  foodInput.addEventListener('keydown', (e)=>{
    const opts = Array.from(typeahead.querySelectorAll('[role=option]'));
    const i = opts.findIndex(o=>o.classList.contains('active'));
    if (e.key === 'ArrowDown') { e.preventDefault(); if(opts.length) { opts.forEach(o=>o.classList.remove('active')); opts[(i+1)%opts.length].classList.add('active'); } }
    if (e.key === 'ArrowUp') { e.preventDefault(); if(opts.length) { opts.forEach(o=>o.classList.remove('active')); opts[(i-1+opts.length)%opts.length].classList.add('active'); } }
    if (e.key === 'Enter') { const a = opts.find(o=>o.classList.contains('active')); if (a){ a.click(); e.preventDefault(); } }
  });

  // Add form submit
  addForm.addEventListener('submit', async (e)=>{
    e.preventDefault();
    addValidation.textContent = '';
    try{
      const data = new FormData(addForm);
      const foodName = data.get('food');
      const qtyValue = Number(data.get('qtyValue'));
      const qtyUnit = data.get('qtyUnit');
      const location = data.get('location');
      const bestBy = data.get('bestBy') || null;
      const opened = data.get('opened') === 'on';
      const notes = data.get('notes');
      if (!foodName || !isFinite(qtyValue) || qtyValue < 0) throw new Error('Please provide a valid food and quantity');
      await FreezerDB.addItem({ foodName, qtyValue, qtyUnit, location, bestBy, opened, notes });
      addForm.reset();
      typeahead.innerHTML = '';
      foodInput.focus();
      await refresh();
      toast('Item added');
    }catch(err){
      addValidation.textContent = err.message || 'Failed to add item';
      toast('Error: ' + (err.message || 'add failed'));
    }
  });

  // Filters
  [searchInput, locFilter, openedOnly, expFilter].forEach(el => el.addEventListener('input', refresh));

  // Scroll to add
  scrollToAdd.addEventListener('click', ()=>{
    $('#addSection').scrollIntoView({ behavior:'smooth', block:'start' });
    foodInput.focus();
  });

  // Export / Import
  exportBtn.addEventListener('click', async ()=>{
    try{
      const data = await FreezerDB.exportAll();
      const blob = new Blob([JSON.stringify(data,null,2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = 'freezer-inventory.json';
      a.click();
      URL.revokeObjectURL(url);
    }catch(err){ toast('Export failed'); }
  });
  importFile.addEventListener('change', async ()=>{
    const file = importFile.files[0];
    if (!file) return;
    try{
      const text = await file.text();
      const json = JSON.parse(text);
      await FreezerDB.importAll(json);
      await refresh();
      toast('Import complete');
    }catch(err){ toast('Import failed'); }
    finally{ importFile.value = ''; }
  });

  // Sample data
  loadSamplesBtn.addEventListener('click', async ()=>{
    await FreezerDB.sampleData();
    await refresh();
    toast('Sample data loaded');
  });

  // Rendering
  async function refresh() {
    foodsCache = await FreezerDB.db.foods.toArray();
    await renderLocations();
    const query = searchInput.value;
    const location = locFilter.value || '';
    const opened = openedOnly.checked;
    const expDays = expFilter.value || '';
    const items = await FreezerDB.searchItems({ query, location, openedOnly: opened, expDays });
    const foodsMap = await FreezerDB.getFoodsMap();
    if (!items.length) {
      emptyEl.hidden = false;
      listEl.innerHTML = '';
      return;
    }
    emptyEl.hidden = true;
    // Group by location
    const groups = new Map();
    items.forEach(it => {
      const loc = it.location || 'Unsorted';
      if (!groups.has(loc)) groups.set(loc, []);
      groups.get(loc).push(it);
    });
    listEl.innerHTML = '';
    [...groups.entries()].sort((a,b)=>a[0].localeCompare(b[0])).forEach(([loc, arr])=>{
      const grp = document.createElement('section');
      grp.className = 'group';
      grp.innerHTML = `<h3>${escapeHtml(loc)}</h3><div class="cards"></div>`;
      const cards = grp.querySelector('.cards');
      arr.sort((a,b)=> (a.bestBy||'').localeCompare(b.bestBy||''));
      arr.forEach(item => cards.appendChild(renderCard(item, foodsMap.get(item.foodId))));
      listEl.appendChild(grp);
    });
  }

  async function renderLocations(){
    const cur = locFilter.value;
    const locs = await FreezerDB.listLocations();
    locFilter.innerHTML = '<option value="">All locations</option>' + locs.map(l=>`<option>${escapeHtml(l)}</option>`).join('');
    if (locs.includes(cur)) locFilter.value = cur;
  }

  function renderCard(item, food){
    const card = document.createElement('article');
    card.className = 'card';
    const best = item.bestBy ? `Best-by ${item.bestBy}` : '';
    const qty = FreezerDB.formatQty(item.qtyValue, item.qtyUnit);
    const soon = item.bestBy && daysUntil(item.bestBy) <= 7;
    card.innerHTML = `
      <div>
        <h4>${escapeHtml(food?.name || 'Unknown')}</h4>
        <div class="meta">${qty}${best?` · ${best}`:''}${item.notes?` · ${escapeHtml(item.notes)}`:''}</div>
        <div class="badges">${item.opened?'<span class="badge">Opened</span>':''}${soon?'<span class="badge warn">Expiring soon</span>':''}</div>
      </div>
      <div class="card-actions">
        <button class="small-btn consume" aria-label="Consume">Consume</button>
        <button class="small-btn edit" aria-label="Edit">Edit</button>
      </div>
      <div class="inline-panel" hidden></div>
    `;

    const panel = card.querySelector('.inline-panel');
    const closePanel = ()=>{ panel.hidden = true; panel.innerHTML = ''; };

    card.querySelector('.consume').addEventListener('click', ()=>{
      panel.hidden = false;
      panel.innerHTML = consumePanel(item);
      const val = panel.querySelector('[name=deltaValue]');
      const unit = panel.querySelector('[name=deltaUnit]');
      const warn = panel.querySelector('.validation');
      const onSubmit = async (ev)=>{
        ev.preventDefault();
        warn.textContent = '';
        try{
          const deltaValue = Number(val.value);
          const deltaUnit = unit.value;
          if (!isFinite(deltaValue) || deltaValue <= 0) throw new Error('Enter a positive amount');
          const res = await FreezerDB.consume({ item, deltaValue, deltaUnit });
          if (res.deleted) toast('Item consumed and removed'); else toast('Quantity updated');
          await refresh();
        }catch(err){
          if (err && err.code === 'NO_DENSITY') {
            warn.textContent = 'Cannot convert between mass and volume without density';
          } else {
            warn.textContent = err.message || 'Failed to consume';
          }
        }
      };
      panel.querySelector('form').addEventListener('submit', onSubmit);
      panel.querySelector('.cancel').addEventListener('click', (e)=>{ e.preventDefault(); closePanel(); });
      panel.addEventListener('keydown', (e)=>{ if (e.key==='Escape'){ closePanel(); card.querySelector('.consume').focus(); } });
      val.focus();
    });

    card.querySelector('.edit').addEventListener('click', ()=>{
      panel.hidden = false;
      panel.innerHTML = editPanel(item);
      const form = panel.querySelector('form');
      const loc = form.querySelector('[name=location]');
      const opened = form.querySelector('[name=opened]');
      const notes = form.querySelector('[name=notes]');
      form.addEventListener('submit', async (e)=>{
        e.preventDefault();
        await FreezerDB.updateItem(item.id, {
          location: loc.value.trim() || 'Unsorted',
          opened: opened.checked,
          notes: notes.value.trim() || null
        });
        toast('Item updated');
        await refresh();
      });
      form.querySelector('.cancel').addEventListener('click', (e)=>{ e.preventDefault(); closePanel(); });
      form.addEventListener('keydown', (e)=>{ if (e.key==='Escape'){ closePanel(); card.querySelector('.edit').focus(); } });
      loc.focus();
    });

    return card;
  }

  function consumePanel(item){
    return `
      <form class="inline-row" aria-label="Consume amount">
        <label class="visually-hidden" for="deltaValue-${item.id}">Amount</label>
        <input id="deltaValue-${item.id}" name="deltaValue" type="number" step="any" min="0" placeholder="Amount" required>
        <label class="visually-hidden" for="deltaUnit-${item.id}">Unit</label>
        <select id="deltaUnit-${item.id}" name="deltaUnit">
          <option value="g">g</option>
          <option value="kg">kg</option>
          <option value="ml">ml</option>
          <option value="L">L</option>
          <option value="count">count</option>
          <option value="serving">serving</option>
        </select>
        <button class="small-btn" type="submit">OK</button>
        <button class="small-btn cancel">Cancel</button>
        <div class="validation" aria-live="polite"></div>
      </form>`;
  }

  function editPanel(item){
    return `
      <form class="inline-row" aria-label="Edit item">
        <label class="visually-hidden" for="loc-${item.id}">Location</label>
        <input id="loc-${item.id}" name="location" type="text" value="${escapeHtml(item.location||'')}" required>
        <label class="checkbox"><input name="opened" type="checkbox" ${item.opened?'checked':''}><span>Opened</span></label>
        <label class="visually-hidden" for="notes-${item.id}">Notes</label>
        <input id="notes-${item.id}" name="notes" type="text" value="${escapeHtml(item.notes||'')}">
        <button class="small-btn" type="submit">Save</button>
        <button class="small-btn cancel">Cancel</button>
      </form>`;
  }

  function daysUntil(iso){
    const today = new Date();
    const d = new Date(iso);
    const A = new Date(today.getFullYear(), today.getMonth(), today.getDate());
    const B = new Date(d.getFullYear(), d.getMonth(), d.getDate());
    return Math.round((B - A) / (1000*60*60*24));
  }

  // Initial load
  (async function init(){
    await refresh();
  })();
})();

