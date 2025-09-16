/* Dexie setup and data helpers */
const DB_NAME = 'freezer-db';
const DB_VERSION = 2;

const db = new Dexie(DB_NAME);
db.version(DB_VERSION).stores({
  foods: '++id,&name,category,densityGPerML,[name+category]',
  items: '++id,foodId,qtyUnit,location,opened,bestBy,addedAt,[foodId+location],location,opened,bestBy'
});

// Utilities
const UnitKind = {
  g: 'mass', kg: 'mass',
  ml: 'volume', L: 'volume',
  count: 'count', serving: 'serving'
};

function toBase(unit, value) {
  const n = Number(value);
  if (!isFinite(n) || n < 0) return null;
  switch(unit){
    case 'kg': return { unit: 'g', value: n * 1000 };
    case 'g': return { unit: 'g', value: n };
    case 'L': return { unit: 'ml', value: n * 1000 };
    case 'ml': return { unit: 'ml', value: n };
    case 'count': return { unit: 'count', value: n };
    case 'serving': return { unit: 'serving', value: n };
    default: return null;
  }
}

function formatQty(value, unit) {
  if (unit === 'g') {
    if (value >= 1000) return `${(value/1000).toFixed(1).replace(/\.0$/,'')} kg`;
    return `${Math.round(value)} g`;
  }
  if (unit === 'ml') {
    if (value >= 1000) return `${(value/1000).toFixed(1).replace(/\.0$/,'')} L`;
    return `${Math.round(value)} ml`;
  }
  if (unit === 'count') return `${value} count`;
  if (unit === 'serving') return `${value} serving${value!==1?'s':''}`;
  return `${value} ${unit}`;
}

function sameKind(u1, u2) {
  return UnitKind[u1] === UnitKind[u2];
}

async function addFoodIfMissing(name) {
  const trimmed = (name||'').trim();
  if (!trimmed) throw new Error('Food name is required');
  // Case-insensitive lookup by name or alias
  const foods = await db.foods.toArray();
  const existing = foods.find(f => f.name.toLowerCase() === trimmed.toLowerCase() || (Array.isArray(f.aliases) && f.aliases.some(a => a.toLowerCase() === trimmed.toLowerCase())));
  if (existing) return existing.id;
  // If a close match exists (startsWith), prefer reuse? Leave to UI for now
  const id = await db.foods.add({ name: trimmed });
  return id;
}

async function addItem({ foodName, qtyValue, qtyUnit, location, bestBy, opened, notes }) {
  const foodId = await addFoodIfMissing(foodName);
  const base = toBase(qtyUnit, qtyValue);
  if (!base) throw new Error('Invalid quantity');
  const nowIso = new Date().toISOString();
  const item = {
    foodId,
    qtyValue: base.value,
    qtyUnit: base.unit,
    location: (location||'').trim() || 'Unsorted',
    addedAt: nowIso,
    bestBy: bestBy || null,
    opened: !!opened,
    notes: (notes||'').trim() || null
  };
  const id = await db.items.add(item);
  return id;
}

async function listLocations() {
  const locs = await db.items.orderBy('location').uniqueKeys();
  return locs.filter(Boolean);
}

async function getFoodsMap() {
  const foods = await db.foods.toArray();
  const map = new Map();
  foods.forEach(f => map.set(f.id, f));
  return map;
}

async function searchItems({ query, location, openedOnly, expDays }) {
  const foods = await db.foods.toArray();
  const q = (query||'').trim().toLowerCase();
  const matchFoodIds = new Set(
    foods.filter(f => {
      if (!q) return true;
      const hay = [f.name, ...(Array.isArray(f.aliases)?f.aliases:[])].join(' ').toLowerCase();
      return hay.includes(q);
    }).map(f => f.id)
  );
  let coll = db.items.where('foodId').anyOf([...matchFoodIds]);
  if (location) coll = coll.and(i => i.location === location);
  if (openedOnly) coll = coll.and(i => !!i.opened);
  if (expDays) {
    const now = new Date();
    const cutoff = new Date(now.getFullYear(), now.getMonth(), now.getDate() + Number(expDays));
    coll = coll.and(i => i.bestBy && new Date(i.bestBy) <= cutoff);
  }
  const items = await coll.sortBy('location');
  return items;
}

async function updateItem(id, patch) {
  await db.items.update(id, patch);
}

async function removeItem(id) {
  await db.items.delete(id);
}

async function consume({ item, deltaValue, deltaUnit }) {
  const baseDelta = toBase(deltaUnit, deltaValue);
  if (!baseDelta) throw new Error('Invalid amount');
  const itemKind = UnitKind[item.qtyUnit];
  const deltaKind = UnitKind[baseDelta.unit];
  if (itemKind !== deltaKind) {
    // Try convert across mass/volume if food has density
    const food = await db.foods.get(item.foodId);
    if (!(food && typeof food.densityGPerML === 'number' && isFinite(food.densityGPerML))) {
      const reason = 'Cross-type conversion requires density';
      const err = new Error(reason);
      err.code = 'NO_DENSITY';
      throw err;
    }
    // Convert delta to item unit using density
    if (deltaKind === 'mass' && itemKind === 'volume') {
      // mass -> volume using density g/ml
      baseDelta.unit = 'ml';
      baseDelta.value = baseDelta.value / food.densityGPerML;
    } else if (deltaKind === 'volume' && itemKind === 'mass') {
      baseDelta.unit = 'g';
      baseDelta.value = baseDelta.value * food.densityGPerML;
    } else {
      throw new Error('Cannot convert between these units');
    }
  }

  const newVal = Math.max(0, (item.qtyValue || 0) - baseDelta.value);
  const epsilon = 0.0001;
  if (newVal <= epsilon) {
    await removeItem(item.id);
    return { deleted: true };
  } else {
    await updateItem(item.id, { qtyValue: newVal });
    return { deleted: false, qtyValue: newVal, qtyUnit: item.qtyUnit };
  }
}

async function exportAll() {
  const [foods, items] = await Promise.all([
    db.foods.toArray(),
    db.items.toArray()
  ]);
  return { version: 1, exportedAt: new Date().toISOString(), foods, items };
}

async function importAll(json) {
  if (!json || typeof json !== 'object') throw new Error('Invalid file');
  const { foods = [], items = [] } = json;
  await db.transaction('rw', db.foods, db.items, async () => {
    if (Array.isArray(foods) && foods.length) await db.foods.bulkPut(foods);
    if (Array.isArray(items) && items.length) await db.items.bulkPut(items);
  });
}

async function sampleData() {
  const ids = {};
  ids.chicken = await addFoodIfMissing('Chicken thighs');
  ids.peas = await addFoodIfMissing('Frozen peas');
  ids.soup = await addFoodIfMissing('Tomato soup');
  ids.bread = await addFoodIfMissing('Sourdough bread');
  ids.pierogi = await addFoodIfMissing('Pierogi');

  const today = new Date();
  const fmt = d => d.toISOString().slice(0,10);
  await addItem({ foodName:'Chicken thighs', qtyValue:2, qtyUnit:'kg', location:'Bin A', bestBy:'2025-12-15', opened:false, notes:'' });
  await addItem({ foodName:'Frozen peas', qtyValue:1.5, qtyUnit:'kg', location:'Bin B', bestBy:fmt(new Date(today.getFullYear(),today.getMonth(),today.getDate()+45)), opened:false, notes:'' });
  await addItem({ foodName:'Tomato soup', qtyValue:1, qtyUnit:'L', location:'Door', bestBy:fmt(new Date(today.getFullYear(),today.getMonth(),today.getDate()+10)), opened:true, notes:'Homemade' });
  await addItem({ foodName:'Sourdough bread', qtyValue:1, qtyUnit:'count', location:'Top shelf', bestBy:fmt(new Date(today.getFullYear(),today.getMonth(),today.getDate()+5)), opened:false, notes:'Half loaf' });
  await addItem({ foodName:'Pierogi', qtyValue:6, qtyUnit:'serving', location:'Bin C', bestBy:null, opened:false, notes:'' });
}

window.FreezerDB = {
  db,
  toBase, formatQty, sameKind,
  addFoodIfMissing, addItem, listLocations, getFoodsMap,
  searchItems, updateItem, removeItem, consume,
  exportAll, importAll, sampleData
};

