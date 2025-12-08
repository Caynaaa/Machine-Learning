document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const clearBtn = document.getElementById('clear-chat');
    const loadingOverlay = document.getElementById('loading-overlay');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceValue = document.getElementById('confidence-value');
    const modelInfo = document.getElementById('model-info');
    const intentInfo = document.getElementById('intent-info');
    const statusDot = document.getElementById('status-dot');
    const statusText = document.querySelector('.status-txt');  

    // Config
    const API_URL = 'http://localhost:5000/api';
    let isConnected = false;

    // Initialize
    checkBackendStatus();
    setupEventListeners();

    // FUNCTIONS 

    function checkBackendStatus() {
        console.log('Checking backend status...');

        fetch(`${API_URL}/health`)
        .then(response => {
            console.log('Respon status:', response.status, response.status.statusText);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json()
        })
            
        .then(data => {
            console.log('Backend response:', data)

            if (data.status === 'healthy') {
                updateStatus('connected');
                updateModelInfo(data);
            } else {
                updateStatus('disconnected');
            }
        })
        .catch(error => {
            console.error('Backend error:', error);
            updateStatus('error');
        });
    }

    function updateStatus(status) {
        const statusConfig = {
            'connecting': {color: 'gray', text: 'Connecting...'},
            'connected': {color: 'green', text: 'Online'},
            'disconnected': {color: 'red', text: 'Offline'},
            'error': {color: 'orange', text: 'Connection Error'}
        };

        const config = statusConfig[status] || statusConfig.connecting;
        statusDot.style.backgroundColor = config.color;
        statusText.textContent = config.text;
        isConnected = (status === 'connected');
    }
    
    function updateModelInfo(data) {
        modelInfo.textContent = `Model: ${data.intents_count} intents loaded`;
    }

  
    function showLoading(show) {
        if (loadingOverlay) {
            loadingOverlay.style.display = show ? 'flex' : 'none';
        }
    }

    function setupEventListeners() {
        sendBtn.addEventListener('click', sendMessage);

        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        clearBtn.addEventListener('click', clearChat);
        userInput.focus();
    }

    function sendMessage() {
        const message = userInput.value.trim();
        if (!message) return;
        if (!isConnected) {
            alert('ChatBot sedang offline. Coba lagi nanti.');
            return;
        }
        
        // Add user message to chat
        addMessageToChat(message, 'user');
        
        // Clear input
        userInput.value = '';
        
        // Show loading
        showLoading(true);

        // Send to backend - ✅ PERBAIKI TYPO
        fetch(`${API_URL}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json', 
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Add bot response
                addMessageToChat(data.response, 'bot', data);

                // Update confidence indicator
                updateConfidenceIndicator(data.confidence);

                // Update intent info
                intentInfo.textContent = `Intent: ${data.intent}`;

                // If warning, show it separately
                if (data.warning) {
                    addWarningMessage(data.warning);
                }
            } else {
                addMessageToChat(`Error: ${data.error}`, 'bot');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            addMessageToChat('Maaf, ada kesalahan dalam memproses pesanmu.', 'bot');
        })
        .finally(() => {
            showLoading(false);
            scrollToBottom();
        });
    }

    function addMessageToChat(text, sender, data = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;

        const time = new Date().toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});

        if (sender === 'bot') {
            messageDiv.innerHTML = `
                <div class="message-avatar">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="message-text">${escapeHtml(text)}</div>
                    <div class="message-time">${time}</div>
                    ${data ? `<div class="message-meta"><span class="confidence-badge">${Math.round(data.confidence * 100)}% confident</span></div>` : ''}
                </div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="message-content">
                    <div class="message-text">${escapeHtml(text)}</div>
                    <div class="message-time">${time}</div>
                </div>
            `;
        }
        chatMessages.appendChild(messageDiv);
        scrollToBottom();
    }

    function addWarningMessage(warning) {
        const warningDiv = document.createElement('div');
        warningDiv.className = 'message warning-message';
        warningDiv.innerHTML = `
            <div class="message-content">
                <div class="warning-text">
                    <i class="fas fa-exclamation-triangle"></i>
                    ${escapeHtml(warning)}
                </div>
            </div>
        `;
        chatMessages.appendChild(warningDiv);
    } 

    function updateConfidenceIndicator(confidence) {
        const percentage = Math.round(confidence * 100);
        confidenceFill.style.width = `${percentage}%`;
        confidenceValue.textContent = `${percentage}%`;

        // Color based on confidence
        if (percentage >= 80) {
            confidenceFill.style.backgroundColor = '#4CAF50'; // Green
        } else if (percentage >= 60) {
            confidenceFill.style.backgroundColor = '#FF9800'; // Orange
        } else {
            confidenceFill.style.backgroundColor = '#F44336'; // Red
        }
    }

    function clearChat() {
        if (confirm('Bersihkan semua pesan chat?')) {
            // Keep only welcome message - ✅ PERBAIKI SCOPE VARIABLE
            const welcomeMsg = chatMessages.querySelector('.bot-message'); 
            
            chatMessages.innerHTML = '';
            
            if (welcomeMsg) {
                chatMessages.appendChild(welcomeMsg);
            }

            // Reset indicators
            confidenceFill.style.width = '0%';
            confidenceValue.textContent = '--%';
            intentInfo.textContent = 'Intent: ---';
        }
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // Auto check status every 30s
    setInterval(checkBackendStatus, 30000);
});