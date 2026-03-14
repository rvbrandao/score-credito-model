const chatPanel = document.getElementById("chatPanel");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const resetButton = document.getElementById("resetButton");
const apiUrlInput = document.getElementById("apiUrl");

let conversationId = null;

function addMessage(role, text) {
  const box = document.createElement("article");
  box.className = `msg ${role}`;
  box.textContent = text;
  chatPanel.appendChild(box);
  chatPanel.scrollTop = chatPanel.scrollHeight;
}

async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text) {
    return;
  }

  const baseUrl = apiUrlInput.value.trim().replace(/\/$/, "");
  if (!baseUrl) {
    addMessage("bot", "Defina a URL da API antes de enviar a mensagem.");
    return;
  }

  addMessage("user", text);
  messageInput.value = "";

  const payload = {
    message: text,
    conversation_id: conversationId,
  };

  try {
    const response = await fetch(`${baseUrl}/score/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const body = await response.json();

    if (!response.ok) {
      const detail = body?.detail || "Erro desconhecido na API.";
      addMessage("bot", `Erro ${response.status}: ${detail}`);
      return;
    }

    conversationId = body.conversation_id;

    const scoreText =
      body.probability_default === null || body.probability_default === undefined
        ? "score pendente"
        : `probabilidade_default=${body.probability_default}`;

    const missingText =
      body.missing_fields && body.missing_fields.length
        ? `\nCampos faltantes: ${body.missing_fields.join(", ")}`
        : "";

    const data = body.extracted_data || {};
    const profileText = `Dados extraidos: age=${data.age}, income=${data.income}, number_of_loans=${data.number_of_loans}, payment_delays=${data.payment_delays}`;

    addMessage(
      "bot",
      `Conversa: ${conversationId} | Turno: ${body.turn}\n${scoreText}\n${profileText}\n${body.explanation}${missingText}`
    );
  } catch (err) {
    addMessage("bot", `Falha de conexao com a API: ${err.message}`);
  }
}

function resetConversation() {
  conversationId = null;
  chatPanel.innerHTML = "";
  addMessage(
    "bot",
    "Nova conversa iniciada. Envie um perfil em linguagem natural para calcular o score."
  );
}

sendButton.addEventListener("click", sendMessage);
resetButton.addEventListener("click", resetConversation);
messageInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

resetConversation();
