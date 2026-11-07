function extractComments() {
  const nodes = document.querySelectorAll(
    "ytd-comment-thread-renderer #content-text"
  );
  return Array.from(nodes)
    .map(n => n.innerText.trim())
    .filter(t => t.length > 0);
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.action === "extract") {
    sendResponse({ comments: extractComments() });
  }
  return true;
});