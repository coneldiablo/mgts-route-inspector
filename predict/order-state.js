export function condition(order) {
  const values = Object.values(order.scores || {}).filter(Number.isFinite);
  if (values.length !== 2) return 'Недостаточно данных';
  return Math.max(...values) >= .5 ? 'Признаки повреждения' : 'Видимых дефектов не выявлено';
}

export function recommendation(order) {
  if (order.request && !order.rephoto) return 'Запросить пересъёмку';
  const signals=Object.values(order.signals || {});
  const assigned=signals.filter(s=>s.decision==='confirmed');
  if(assigned.some(s=>s.result!=='resolved')) return 'Выполнить подтверждённые работы';
  if(assigned.length && assigned.length+signals.filter(s=>s.decision==='dismissed').length===signals.length) return 'Работы приняты';
  if(assigned.length) return 'Проверить оставшиеся признаки';
  if(signals.length && signals.every(s=>s.decision==='dismissed')) return 'Мастер снял замечания';
  return condition(order)==='Видимых дефектов не выявлено' ? 'Плановый контроль' : 'Проверить признаки';
}
