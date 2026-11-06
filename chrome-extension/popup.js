const API_URL = "http://localhost:7860"; // GitHub Codespaces (port 7860)
let results = [];
const $ = id => document.getElementById(id);

function escapeHtml(s) {
  return s.replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

$("analyze").onclick = async () => {
  $("status").textContent = "Extraction des commentaires...";
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  try {
    const resp = await chrome.tabs.sendMessage(tab.id, { action: "extract" });
    const comments = resp.comments;
    if (!comments.length) {
      $("status").textContent = "Aucun commentaire trouvé. Scrollez jusqu'aux commentaires puis réessayez.";
      return;
    }
    $("status").textContent = `Analyse de ${comments.length} commentaires...`;
    const res = await fetch(`${API_URL}/predict_batch`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ comments })
    });
    if (!res.ok) throw new Error(`Erreur API ${res.status}`);
    const data = await res.json();
    results = data.predictions;
    renderStats(data.statistics);
    renderList("all");
    $("status").textContent = `Terminé : ${comments.length} commentaires en ${data.processing_time_ms} ms (API)`;
  } catch (e) {
    $("status").textContent = "Erreur : " + e.message + " (page YouTube rechargée ? API lancée ?)";
  }
};

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