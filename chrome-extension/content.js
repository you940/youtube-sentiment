const sleep = ms => new Promise(r => setTimeout(r, ms));

function getComments() {
  const nodes = document.querySelectorAll(
    "ytd-comment-thread-renderer #content-text"
  );
  return Array.from(nodes)
    .map(n => n.innerText.trim())
    .filter(t => t.length > 0);
}

// Fait défiler la page pour que YouTube charge assez de commentaires
async function loadComments(limit) {
  const startY = window.scrollY;
  let last = -1;
  let stable = 0;
  let loops = 0;

  while (getComments().length < limit && stable < 6 && loops < 500) {
    const threads = document.querySelectorAll("ytd-comment-thread-renderer");
    const target = threads.length
      ? threads[threads.length - 1]
      : document.querySelector("ytd-comments");
    if (target) target.scrollIntoView({ block: "end" });
    else window.scrollTo(0, document.documentElement.scrollHeight);
    window.scrollBy(0, 600);

    await sleep(1000);
    const n = getComments().length;
    stable = n === last ? stable + 1 : 0;   // plus rien de nouveau => fin des commentaires
    last = n;
    loops++;
  }

  window.scrollTo(0, startY);
  return getComments().slice(0, limit);
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "extract") {
    loadComments(msg.limit || 200).then(comments => sendResponse({ comments }));
    return true; // réponse asynchrone
  }
});