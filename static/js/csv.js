// @ts-nocheck
(function () {
  'use strict';

  var API = window.API_BASE || 'http://127.0.0.1:8000';

  function renderTable(data) {
    var cols = [];
    if (Array.isArray(data) && data.length) {
      Object.keys(data[0]).forEach(function (k) {
        cols.push({ title: k, data: k });
      });
    }

    if (!window.$ || !$.fn || !$.fn.DataTable) {
      console.error('DataTables が読み込まれていません（jQuery → DataTables → Buttonsの順で読み込んでください）');
      return;
    }

    if ($.fn.dataTable.isDataTable('#csvTable')) {
      var inst = $('#csvTable').DataTable();
      inst.clear().rows.add(data || []).draw(false);
      return;
    }

    $('#csvTable').DataTable({
      data: data || [],
      columns: cols,
      pageLength: 50,
      order: [[0, 'desc']],
      dom: 'Bfrtip',
      buttons: [{ extend: 'csvHtml5', text: 'CSV ダウンロード', title: 'umi-no-me-detections' }],
      language: { url: 'https://cdn.datatables.net/plug-ins/1.13.7/i18n/ja.json' }
    });
  }

  function fetchAndRender() {
    fetch(API + '/csv-data')
      .then(function (res) {
        if (!res.ok) throw new Error(res.status + ' ' + res.statusText);
        return res.json();
      })
      .then(function (json) { renderTable(json); })
      .catch(function (err) { console.error('csv fetch error:', err); });
  }

  function boot() {
    fetchAndRender();
    setInterval(fetchAndRender, 10000); // 10秒毎に更新（任意）
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot, { once: true });
  } else {
    boot();
  }
})();
