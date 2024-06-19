async function askQuestion() {
    const prompt = document.getElementById('prompt').value;
    if (!prompt) return;

    addMessage(prompt, 'user-message');
    document.getElementById('prompt').value = '';

    const response = await fetch('http://127.0.0.1:5000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt })
    });

    const data = await response.json();
    addMessage(data.response, 'assistant-message');
}

function addMessage(message, className) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${className}`;
    messageDiv.innerText = message;
    document.getElementById('messages').appendChild(messageDiv);
    messageDiv.scrollIntoView();
}
