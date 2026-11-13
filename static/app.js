// Weather Advisory Bot Client Logic

let currentSessionId = localStorage.getItem('weather_session_id') || generateUUID();
localStorage.setItem('weather_session_id', currentSessionId);

const messagesContainer = document.getElementById('messagesContainer');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const sessionBadge = document.getElementById('sessionBadge');
const resetBtn = document.getElementById('resetBtn');
const errorBanner = document.getElementById('errorBanner');

function generateUUID() {
  return 'sess-' + Math.random().toString(36).substring(2, 10) + '-' + Date.now().toString(36);
}

function updateSessionUI() {
  sessionBadge.textContent = 'Session: ' + currentSessionId.substring(0, 14) + '...';
  sessionBadge.title = 'Full Session ID: ' + currentSessionId;
}

updateSessionUI();

resetBtn.addEventListener('click', () => {
  currentSessionId = generateUUID();
  localStorage.setItem('weather_session_id', currentSessionId);
  updateSessionUI();
  messagesContainer.innerHTML = `
    <div class="message bot-message intro-message">
      <div class="message-content">
        <p><strong>Session reset.</strong> Previous conversation context has been cleared.</p>
        <p>Ask a new outdoor activity question!</p>
        <div class="quick-prompts">
          <button class="chip" onclick="usePrompt('Is it okay to cycle to work in Bhopal today?')">🚴 Cycle in Bhopal</button>
          <button class="chip" onclick="usePrompt('Can we have an outdoor picnic in London today?')">🧺 Picnic in London</button>
          <button class="chip" onclick="usePrompt('Is it safe for crane and scaffolding construction in Tokyo?')">🏗️ Construction in Tokyo</button>
        </div>
      </div>
    </div>
  `;
  hideError();
});

function usePrompt(text) {
  messageInput.value = text;
  messageInput.focus();
}

function showError(msg) {
  errorBanner.textContent = msg;
  errorBanner.classList.remove('hidden');
}

function hideError() {
  errorBanner.textContent = '';
  errorBanner.classList.add('hidden');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function renderMarkdown(md) {
  if (!md) return '';
  let html = escapeHtml(md);

  // Headers
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^#### (.*$)/gim, '<h4>$1</h4>');

  // Bold
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Inline code
  html = html.replace(/`(.*?)`/g, '<code>$1</code>');

  // Blockquotes
  html = html.replace(/^> (.*$)/gim, '<blockquote>$1</blockquote>');

  // Bullet points
  html = html.replace(/^- (.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

  // Line breaks
  html = html.replace(/\n\n/g, '<br/><br/>');
  html = html.replace(/\n/g, '<br/>');

  return html;
}

function appendMessage(role, content, citation = null) {
  const msgEl = document.createElement('div');
  msgEl.className = `message ${role}-message`;

  let citationHtml = '';
  if (citation && citation.id) {
    const sev = citation.severity || 'Normal';
    citationHtml = `<div class="citation-pill severity-${sev}">Policy Cited: [${citation.id}] ${citation.title} &bull; ${sev}</div>`;
  }

  const parsedContent = role === 'bot' ? renderMarkdown(content) : escapeHtml(content);

  msgEl.innerHTML = `
    <div class="message-content">
      ${citationHtml}
      ${parsedContent}
    </div>
  `;

  messagesContainer.appendChild(msgEl);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function showLoadingIndicator() {
  const loadingEl = document.createElement('div');
  loadingEl.id = 'typingIndicator';
  loadingEl.className = 'message bot-message';
  loadingEl.innerHTML = `
    <div class="typing-indicator">
      <div class="dot"></div>
      <div class="dot"></div>
      <div class="dot"></div>
    </div>
  `;
  messagesContainer.appendChild(loadingEl);
  messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeLoadingIndicator() {
  const el = document.getElementById('typingIndicator');
  if (el) el.remove();
}

// Handle auto-expanding textarea and Enter key
messageInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event('submit'));
  }
});

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;

  hideError();
  appendMessage('user', text);
  messageInput.value = '';
  messageInput.disabled = true;
  sendBtn.disabled = true;
  showLoadingIndicator();

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: currentSessionId,
        message: text,
      }),
    });

    removeLoadingIndicator();

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      const errMsg = errData.detail || `Server returned error status HTTP ${response.status}`;
      showError(errMsg);
      appendMessage('bot', '**Error communicating with server:** ${errMsg}');
      return;
    }

    const data = await response.json();
    if (data.session_id) {
      currentSessionId = data.session_id;
      localStorage.setItem('weather_session_id', currentSessionId);
      updateSessionUI();
    }

    appendMessage('bot', data.answer, data.sop_citation);

  } catch (err) {
    removeLoadingIndicator();
    showError(`Network connection failure: ${err.message}`);
    appendMessage('bot', '**Network failure:** Could not reach backend server at /chat.');
  } finally {
    messageInput.disabled = false;
    sendBtn.disabled = false;
    messageInput.focus();
  }
});
