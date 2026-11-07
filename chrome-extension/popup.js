const API_URL = "https://improved-space-xylophone-5g5pxqr755r5h4gwx-7860.app.github.dev";
const CHUNK = 500; // limite de l'API par requête
let results = [];
const $ = id => document.getElementById(id);

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

$("analyze").onclick = async () => {
  let limit = parseInt($("limit").value, 10);
  if (!limit || limit < 1) limit = 200;

  $("status").textContent = `Chargement des commentaires (jusqu'à ${limit}), ne fermez pas le popup...`;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  try {
    const resp = await chrome.tabs.sendMessage(tab.id, { action: "extract", limit });
    const comments = resp.comments;
    if (!comments.length) {
      $("status").textContent = "Aucun commentaire trouvé. Scrollez jusqu'aux commentaires puis réessayez.";
      return;
    }

    results = [];
    let totalMs = 0;
    for (let i = 0; i < comments.length; i += CHUNK) {
      const part = comments.slice(i, i + CHUNK);
      $("status").textContent = `Analyse de ${Math.min(i + CHUNK, comments.length)}/${comments.length} commentaires...`;
      const res = await fetch(`${API_URL}/predict_batch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comments: part })
      });
      if (!res.ok) throw new Error(`Erreur API ${res.status}`);
      const data = await res.json();
      results = results.concat(data.predictions);
      totalMs += data.processing_time_ms;
    }

    renderStats(computeStats(results));
    renderList("all");
    const note = comments.length < limit ? ` (la vidéo n'en a que ${comments.length})` : "";
    $("status").textContent = `Terminé : ${comments.length} commentaires en ${totalMs.toFixed(2)} ms (API)${note}`;
  } catch (e) {
    $("status").textContent = "Erreur : " + e.message + " (page YouTube rechargée ? API lancée ?)";
  }
};

function computeStats(preds) {
  const names = { "1": "positif", "0": "neutre", "-1": "négatif" };
  const counts = { "positif": 0, "neutre": 0, "négatif": 0 };
  preds.forEach(p => counts[names[p.sentiment]]++);
  const n = preds.length;
  const percentages = {};
  for (const k in counts) percentages[k] = Math.round(1000 * counts[k] / n) / 10;
  return { total: n, counts, percentages };
}

function renderStats(s) {
  const p = s.percentages;
  $("stats").innerHTML = ["positif", "neutre", "négatif"].map(k => `
    <div class="bar-row">${k} : ${p[k]}% (${s.counts[k]})
      <div class="bar"><div class="fill ${k}" style="width:${p[k]}%"></div></div>
    </div>`).join("");
}

function renderList(filter) {
  const items = results.filter(r => filter === "all" || r.sentiment == filter);
  $("list").innerHTML = items.map(r =>
    `<li class="s${r.sentiment}">${escapeHtml(r.text.slice(0, 150))}
      <small>${r.label} (${Math.round(r.confidence * 100)}%)</small></li>`).join("");
}

document.querySelectorAll("#filters button").forEach(b => b.onclick = () => {
  document.querySelectorAll("#filters button").forEach(x => x.classList.remove("active"));
  b.classList.add("active");
  renderList(b.dataset.f);
});

$("theme").onclick = () => {
  document.body.classList.toggle("dark");
  chrome.storage.local.set({ dark: document.body.classList.contains("dark") });
};
chrome.storage.local.get("dark", ({ dark }) => dark && document.body.classList.add("dark"));

$("copy").onclick = () => {
  const txt = results.map(r => `[${r.label} ${Math.round(r.confidence * 100)}%] ${r.text}`).join("\n");
  navigator.clipboard.writeText(txt);
  $("status").textContent = "Résultats copiés dans le presse-papiers !";
};