const API_URL = '/v1/chat/completions';
const MAX_FILE_SIZE = 25 * 1024 * 1024;
const STORAGE_KEY = 'brick_chat_history';

class ChatApp {
    constructor() {
        this.attachments = [];
        this.messages = [];
        this.recorder = new AudioRecorder();
        this.isRecording = false;
        this.typingElement = null;
        this.isProcessing = false;
        this.isStreaming = localStorage.getItem('stream_enabled') !== 'false';
        this.routingAnimationStartTime = null;
        this.minRoutingAnimationDuration = 1000;
        this.routingAnimationActive = false;
        this.routingMessageElement = null;

        this.init();
    }

    init() {
        this.bindElements();
        this.bindEvents();
        this.loadHistory();
        // Ensure a persistent placeholder for assistant messages / routing animation
        if (!document.getElementById('assistantPlaceholder')) {
            const placeholder = document.createElement('div');
            placeholder.id = 'assistantPlaceholder';
            placeholder.className = 'message regolo-brick';
            placeholder.style.display = 'none';
            this.elements.messages.appendChild(placeholder);
        }
        this.assistantPlaceholder = document.getElementById('assistantPlaceholder');
    }

     bindElements() {
         this.elements = {
             messages: document.getElementById('messages'),
             input: document.getElementById('messageInput'),
             fileInput: document.getElementById('fileInput'),
             attachBtn: document.getElementById('attachBtn'),
             micBtn: document.getElementById('micBtn'),
             sendBtn: document.getElementById('sendBtn'),
             clearBtn: document.getElementById('clearBtn'),
             attachments: document.getElementById('attachmentsPreview'),
             routingBtn: document.getElementById('routingBtn'),
             settingsPopup: document.getElementById('settingsPopup'),
             routingPopup: document.getElementById('routingPopup'),
             streamToggle: document.getElementById('streamToggle')
         };
         this.routingMessageElement = null;
         this.routingPopupElement = this.elements.routingPopup;
     }

    bindEvents() {
        this.elements.input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.send();
            }
        });

        this.elements.sendBtn.addEventListener('click', () => this.send());
        this.elements.attachBtn.addEventListener('click', () => {
            this.elements.fileInput.click();
        });

        this.elements.fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
            e.target.value = '';
        });

        this.elements.micBtn.addEventListener('click', () => this.toggleRecording());
        this.elements.clearBtn.addEventListener('click', () => this.clearChat());

        document.addEventListener('click', (e) => {
            if (!e.target.closest('.modality-badge')) {
                document.querySelectorAll('.modality-dropdown').forEach(d => {
                    d.classList.remove('show');
                });
            }
            if (!e.target.closest('#settingsPopup') && !e.target.closest('#routingBtn')) {
                this.closeSettingsPopup();
            }
        });

        this.handleMobileViewport();
        window.addEventListener('resize', () => this.handleMobileViewport());

        this.elements.input.addEventListener('focus', () => {
            document.body.classList.add('input-focused');
        });

        this.elements.input.addEventListener('blur', () => {
            document.body.classList.remove('input-focused');
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeSettingsPopup();
                document.querySelectorAll('.modality-dropdown').forEach(d => {
                    d.classList.remove('show');
                });
            }
        });

        this.elements.streamToggle.checked = this.isStreaming;
        this.elements.streamToggle.addEventListener('change', (e) => {
            this.isStreaming = e.target.checked;
            localStorage.setItem('stream_enabled', this.isStreaming);
        });

         this.elements.routingBtn.addEventListener('click', (e) => {
             e.stopPropagation();
             this.toggleRoutingPopup();
         });
     }

     toggleRoutingPopup() {
         const popup = this.elements.routingPopup;
         if (!popup) return;
         
         const isShowing = popup.classList.contains('show');
         
         if (isShowing) {
             this.closeRoutingPopup();
         } else {
             this.closeSettingsPopup();
             popup.classList.add('show');
         }
     }

    handleMobileViewport() {
        const vh = window.innerHeight * 0.01;
        document.documentElement.style.setProperty('--vh', `${vh}px`);

        const isMobile = window.innerWidth <= 428;
        if (isMobile) {
            document.body.classList.add('mobile-device');
        } else {
            document.body.classList.remove('mobile-device');
        }
    }

    async toggleRecording() {
        const btn = this.elements.micBtn;

        if (!this.isRecording) {
            try {
                await this.recorder.start();
                btn.classList.add('recording');
                this.isRecording = true;
            } catch (error) {
                this.addMessage('system', 'microphone access denied', 'error');
            }
        } else {
            try {
                const base64 = await this.recorder.stop();
                this.attachments.push({
                    type: 'audio',
                    data: base64,
                    name: `audio_${Date.now()}.webm`,
                    icon: 'snd'
                });
                this.renderAttachments();
                btn.classList.remove('recording');
                this.isRecording = false;
            } catch (error) {
                btn.classList.remove('recording');
                this.isRecording = false;
            }
        }
    }

    handleFiles(files) {
        Array.from(files).forEach(async (file, index) => {
            if (file.size > MAX_FILE_SIZE) {
                this.addMessage('system', `file too large (max 25MB)`, 'error');
                return;
            }

            try {
                const base64 = await AudioRecorder.fileToBase64(file);
                const isImage = file.type.startsWith('image/');

                this.attachments.push({
                    type: isImage ? 'image' : 'audio',
                    data: base64,
                    name: file.name,
                    icon: isImage ? 'img' : 'snd',
                    preview: isImage ? base64 : null
                });

                this.renderAttachments();
            } catch (error) {
                this.addMessage('system', `error reading file`, 'error');
            }
        });
    }

    renderAttachments() {
        this.elements.attachments.innerHTML = this.attachments.map((att, i) => `
            <div class="attachment-item">
                ${att.preview ? `<img src="${att.preview}" alt="${att.name}">` : `<span>${att.icon}</span>`}
                <span class="name" title="${att.name}">${att.name}</span>
                <span class="remove" onclick="app.removeAttachment(${i})">×</span>
            </div>
        `).join('');
    }

    removeAttachment(index) {
        this.attachments.splice(index, 1);
        this.renderAttachments();
    }

    buildPayload() {
        const text = this.elements.input.value.trim();

        const content = [];

        if (text) {
            content.push({ type: 'text', text });
        }

        this.attachments.forEach(att => {
            if (att.type === 'image') {
                content.push({
                    type: 'image_url',
                    image_url: { url: att.data }
                });
            } else if (att.type === 'audio') {
                content.push({
                    type: 'input_audio',
                    audio_url: { url: att.data }
                });
            }
        });

        return {
            model: 'brick',
            messages: [{ role: 'user', content }]
        };
    }

    detectModality(content) {
        const hasText = content.some(c => c.type === 'text');
        const hasImage = content.some(c => c.type === 'image_url');
        const hasAudio = content.some(c => c.type === 'input_audio');

        if (hasImage && hasAudio) return '📋 Full Mixed';
        if (hasImage && hasText) return '🖼️ Vision';
        if (hasImage) return '🖼️ OCR';
        if (hasAudio && hasText) return '🎙️ → LLM';
        if (hasAudio) return '🎙️ STT';
        return '🔤 Text';
    }

    formatContent(content) {
        return content.map(c => {
            if (c.type === 'text') return c.text;
            if (c.type === 'image_url') return '[🖼️ Image]';
            if (c.type === 'input_audio') return '[🎵 Audio]';
            return '';
        }).filter(Boolean).join('\n');
    }

     async send() {
         const text = this.elements.input.value.trim();
         if (!text && this.attachments.length === 0) return;
         if (this.isProcessing) return;

         const payload = this.buildPayload();
         payload.stream = this.isStreaming;
         const modality = this.detectModality(payload.messages[0].content);
         const displayContent = this.formatContent(payload.messages[0].content);

         console.log('=== SEND START ===');
         console.log('Text:', text);
         console.log('Attachments:', this.attachments);
         console.log('Payload:', payload);
         console.log('Modality:', modality);
         console.log('Streaming:', this.isStreaming);

          this.addMessage('you', displayContent, modality);

          const isTextRequest = modality === '🔤 Text' || modality === '🎙️ → LLM';

          try {
          this.isProcessing = true;
              // Show routing animation for every request (min 1 second enforced by hideRoutingAnimation)
              this.showRoutingAnimation();
              // No typing indicator needed; routing animation will be shown instead
              console.log('Sending request to:', API_URL);

              console.log('Request body:', JSON.stringify(payload));

              const response = await fetch(API_URL, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify(payload)
              });

              console.log('Response status:', response.status, response.statusText);
              console.log('Response headers:', Object.fromEntries(response.headers.entries()));

              // No need to hide routing animation here; it will be hidden after we get the response.


             if (!response.ok) {
                 const errorText = await response.text();
                 console.error('Error response text:', errorText);
                 let errorMessage = `Error: ${response.status}`;
                 try {
                     const errorJson = JSON.parse(errorText);
                     errorMessage = errorJson.detail || errorMessage;
                 } catch (e) {
                     console.error('Failed to parse error JSON:', e);
                 }
                 console.error('Throwing error:', errorMessage);
                 throw new Error(errorMessage);
             }

if (payload.stream && response.body) {
                  await this.handleStreamingResponse(response, payload);
              } else {
                 const result = await response.json();
                 console.log('Response:', result);
                 console.log('First choice structure:', JSON.stringify(result.choices[0], null, 2));

if (result.choices && result.choices.length > 0) {
                      // Hide routing animation (ensuring the minimum display time of 1 s)
this.hideRoutingAnimation(true);
                      const brickContent = result.choices[0].message.content;
                      console.log('Brick content length:', brickContent?.length);
                      const responseModality = this.detectModality(payload.messages[0].content);
                      const modelsUsed = result.models_used || [result.model];
                      console.log('Models used:', modelsUsed);
                      this.addMessage('regolo-brick', brickContent, responseModality, null, modelsUsed);
                      console.log('=== SEND SUCCESS ===');
                  } else {
                     console.error('Invalid response format:', result);
                     throw new Error('invalid response');
                 }
             }

         } catch (error) {
             console.error('=== SEND ERROR ===');
             console.error('Error object:', error);
             console.error('Error message:', error.message);
             console.error('Error stack:', error.stack);

this.hideRoutingAnimation();
             this.addMessage('regolo-brick', `error: ${error.message}`, null, 'error');
         } finally {
             this.elements.input.value = '';
             this.attachments = [];
             this.renderAttachments();

             this.isProcessing = false;
         }
     }

    async handleStreamingResponse(response, payload) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let assistantMessage = null;
        let messageDiv = null;

        const responseModality = this.detectModality(payload.messages[0].content);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.slice(6);
                    if (data === '[DONE]') continue;

                    try {
                        const chunk = JSON.parse(data);
                        const content = this.parseStreamingChunk(chunk);

                        if (content !== null) {
                        if (!assistantMessage) {
                            // Hide the routing animation now that streaming data is arriving
                            this.hideRoutingAnimation(true);
                            assistantMessage = { sender: 'regolo-brick', content: '', modality: responseModality };
                            this.messages.push(assistantMessage);
                            messageDiv = this.renderStreamingMessage(assistantMessage);
                        }

                            assistantMessage.content += content;
                            this.updateStreamingMessage(messageDiv, assistantMessage.content);
                        }
                    } catch (e) {
                        console.warn('Failed to parse streaming chunk:', e);
                    }
                }
            }
        }

        if (assistantMessage && assistantMessage.content) {
            const modelsUsed = this.extractModelsFromLastChunk();
            this.updateMessageMetadata(messageDiv, assistantMessage.content, modelsUsed);
            this.saveHistory();
            console.log('=== SEND SUCCESS (STREAMING) ===');
        } else {
            throw new Error('No content received in streaming response');
        }
    }

    parseStreamingChunk(chunk) {
        if (chunk.choices && chunk.choices.length > 0) {
            const delta = chunk.choices[0].delta;
            if (delta && delta.content) {
                return delta.content;
            }
        }
        if (chunk.choices && chunk.choices.length > 0) {
            const content = chunk.choices[0].message?.content;
            if (content) return content;
        }
        return null;
    }

    extractModelsFromLastChunk() {
        return null;
    }

    renderStreamingMessage(message) {
        const placeholder = this.assistantPlaceholder;
        placeholder.innerHTML = `
            <div class="message-header">
                <span class="sender">${message.sender}</span>
            </div>
            <div class="message-content streaming"></div>
        `;
        placeholder.style.display = 'block';
        this.scrollToBottom();
        return placeholder;
    }

    updateStreamingMessage(div, content) {
        const contentDiv = div.querySelector('.message-content');
        contentDiv.innerHTML = marked.parse(content);
        this.scrollToBottom();
        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([div]).catch(err => console.warn('MathJax rendering error:', err));
        }
    }

    updateMessageMetadata(div, content, modelsUsed) {
        const header = div.querySelector('.message-header');
        div.classList.remove('streaming');

        const modelsList = modelsUsed ? modelsUsed.map(m => {
            const logo = this.getModelLogo(m);
            const logoHtml = logo ? `<img src="/logos/${logo}" alt="${m}">` : '<span style="width:24px;height:24px;display:inline-block;"></span>';
            return `<li>${logoHtml}<span>${m}</span></li>`;
        }).join('') : '';

        header.innerHTML = `
            <span class="sender">${div.querySelector('.sender').textContent}</span>
            <span class="modality-badge">
                ${div.querySelector('.sender').textContent === 'regolo-brick' ? 'Streaming Complete' : ''}
            </span>
        `;

        div.querySelector('.message-content').innerHTML = marked.parse(content);
        div.dataset.content = content;

        div.innerHTML += `
            <div class="message-actions">
                <span class="model-badge">regolo/brick</span>
                <button class="message-action-btn copy-btn" onclick="app.copyMessage(this)" title="Copy">
                    ${ICONS.copy}
                </button>
                <button class="message-action-btn refresh-btn" onclick="app.regenerateMessage(this)" title="Regenerate">
                    ${ICONS.refresh}
                </button>
            </div>
        `;

        this.scrollToBottom();
    }


    addMessage(sender, content, modality = null, type = null, modelsUsed = null) {
        const message = {
            sender,
            content,
            modality,
            type,
            modelsUsed,
            timestamp: Date.now()
        };

        this.messages.push(message);
        this.renderMessage(message);
        this.saveHistory();
    }

    getModelLogo(modelName) {
        const logos = {
            'gpt-oss-120b': 'openailogo.jpg',
            'gpt-oss-20b': 'openailogo.jpg',
            'deepseek-ocr': 'deepseeklogo.png',
            'deepseek-r1-70b': 'deepseeklogo.png',
            'faster-whisper-large-v3': 'openailogo.jpg',
            'qwen3-vl-32b': 'qwenlogo.png',
            'qwen3-8b': 'qwenlogo.png',
            'qwen3-30b': 'qwenlogo.png',
            'qwen3-coder-30b': 'qwenlogo.png',
            'llama-3.1-8b': 'llamalogo.png',
            'llama-3.3-70b': 'llamalogo.png',
            'gemma-3-27b': 'gemmalogo.png',
            'mistral-small3.2': 'mistrallogo.png'
        };
        return logos[modelName.toLowerCase()] || null;
    }

    getModelsFromModality(modality) {
        const models = {
            '🔤 Text': ['gpt-oss-120b'],
            '🖼️ OCR': ['deepseek-ocr'],
            '🎙️ STT': ['faster-whisper-large-v3'],
            '🖼️ Vision': ['qwen3-vl-32b'],
            '🎙️ → LLM': ['faster-whisper-large-v3', 'gpt-oss-120b'],
            '📋 Full Mixed': ['qwen3-vl-32b', 'faster-whisper-large-v3', 'gpt-oss-120b']
        };
        return models[modality] || ['gpt-oss-120b'];
    }

    renderMessage(message) {
        let targetDiv;
        if (message.sender === 'regolo-brick') {
            // Use persistent placeholder for assistant messages
            targetDiv = this.assistantPlaceholder;
            targetDiv.className = 'message regolo-brick';
            targetDiv.style.display = 'block';
        } else {
            const div = document.createElement('div');
            const cssClass = message.type || (message.sender === 'you' ? 'user' : 'regolo-brick');
            div.className = `message ${cssClass}`;
            targetDiv = div;
        }

        let html = `
            <div class="message-header">
                <span class="sender">${message.sender}</span>
        `;

        if (message.modelsUsed && message.modelsUsed.length > 0) {
            const modelsList = message.modelsUsed.map(m => {
                const logo = this.getModelLogo(m);
                const logoHtml = logo ? `<img src="/logos/${logo}" alt="${m}">` : '<span style="width:24px;height:24px;display:inline-block;"></span>';
                return `<li>${logoHtml}<span>${m}</span></li>`;
            }).join('');
            html += `
                <span class="modality-badge" onclick="app.toggleDropdown(this)">
                    ${message.modality || 'Models Used'}
                    <div class="modality-dropdown">
                        <h4>models used</h4>
                        <ul>${modelsList}</ul>
                    </div>
                </span>
            `;
        } else if (message.modality) {
            const models = this.getModelsFromModality(message.modality);
            const modelsList = models.map(m => {
                const logo = this.getModelLogo(m);
                const logoHtml = logo ? `<img src="/logos/${logo}" alt="${m}">` : '<span style="width:24px;height:24px;display:inline-block;"></span>';
                return `<li>${logoHtml}<span>${m}</span></li>`;
            }).join('');
            html += `
                <span class="modality-badge" onclick="app.toggleDropdown(this)">
                    ${message.modality}
                    <div class="modality-dropdown">
                        <h4>models used</h4>
                        <ul>${modelsList}</ul>
                    </div>
                </span>
            `;
        }

        const renderedContent = message.sender === 'regolo-brick'
            ? marked.parse(message.content)
            : this.escapeHtml(message.content);

        html += `
            </div>
            <div class="message-content">${renderedContent}</div>
        `;

        if (message.sender === 'regolo-brick') {
            html += `
                <div class="message-actions">
                    <span class="model-badge">regolo/brick</span>
                    <button class="message-action-btn copy-btn" onclick="app.copyMessage(this)" title="Copy">
                        ${ICONS.copy}
                    </button>
                    <button class="message-action-btn refresh-btn" onclick="app.regenerateMessage(this)" title="Regenerate">
                        ${ICONS.refresh}
                    </button>
                </div>
            `;
        }

        targetDiv.innerHTML = html;

        if (message.sender === 'regolo-brick') {
            targetDiv.dataset.content = message.content;
        }

        if (targetDiv !== this.assistantPlaceholder) {
            this.elements.messages.appendChild(targetDiv);
        }
        this.scrollToBottom();

        if (window.MathJax && window.MathJax.typesetPromise) {
            window.MathJax.typesetPromise([targetDiv]).catch(err => console.warn('MathJax rendering error:', err));
        }
    }

    copyMessage(btn) {
        const messageDiv = btn.closest('.message');
        const content = messageDiv.dataset.content;
        const originalIcon = ICONS.copy;
        navigator.clipboard.writeText(content).then(() => {
            btn.classList.add('copied');
            btn.innerHTML = ICONS.check;
            setTimeout(() => {
                btn.classList.remove('copied');
                btn.innerHTML = originalIcon;
            }, 2000);
        });
    }

     regenerateMessage(btn) {
         const messageDiv = btn.closest('.message');
         const messages = this.elements.messages;
         const allMessages = Array.from(messages.children);
         const messageIndex = allMessages.indexOf(messageDiv);

         let userMessageIndex = messageIndex - 1;
         while (userMessageIndex >= 0) {
             const prevMessage = allMessages[userMessageIndex];
             if (prevMessage.classList.contains('user')) {
                 messageDiv.remove();
                 const userContent = prevMessage.querySelector('.message-content').textContent;
                 this.elements.input.value = userContent;
                 this.attachments = [];
                 this.renderAttachments();
                 this.send();
                 break;
             }
             userMessageIndex--;
         }
     }

    toggleDropdown(badge) {
        const dropdown = badge.querySelector('.modality-dropdown');
        const isShowing = dropdown.classList.contains('show');

        document.querySelectorAll('.modality-dropdown').forEach(d => {
            d.classList.remove('show');
        });

        if (!isShowing) {
            dropdown.classList.add('show');
        }
    }

    toggleSettingsPopup() {
        const popup = this.elements.settingsPopup;
        const isShowing = popup.classList.contains('show');

        if (isShowing) {
            this.closeSettingsPopup();
        } else {
            popup.classList.add('show');
        }
    }

     closeSettingsPopup() {
         this.elements.settingsPopup.classList.remove('show');
     }

     closeRoutingPopup() {
         if (this.elements.routingPopup) {
             this.elements.routingPopup.classList.remove('show');
         }
     }

    showRoutingAnimation() {
        this.routingAnimationStartTime = Date.now();
        // Use the pre‑existing hidden routingMessage element (from HTML) as a persistent container
        const routingEl = document.getElementById('routingMessage');
        this.routingMessageElement = routingEl;
        const modelsList = [
            { name: 'openai', logo: 'openailogo.jpg' },
            { name: 'google', logo: 'gemmalogo.png' },
            { name: 'meta', logo: 'llamalogo.png' },
            { name: 'deepseek', logo: 'deepseeklogo.png' },
            { name: 'mistral', logo: 'mistrallogo.png' },
            { name: 'qwen', logo: 'qwenlogo.png' }
        ];
        const modelsHtml = modelsList.map(m => `
            <div class=\"routing-message-model\" data-model=\"${m.name}\">
                <img src=\"/logos/${m.logo}\" alt=\"${m.name}\">
            </div>
        `).join('');
        routingEl.innerHTML = `
            <div class=\"routing-message-header\">
                <span class=\"sender\">regolo-brick</span>
            </div>
            <div class=\"routing-message-text\">sorting to the best model in class</div>
            <div class=\"routing-message-models\">
                ${modelsHtml}
            </div>
        `;
        routingEl.style.display = 'block';
        this.scrollToBottom();
        // Start animation on model elements inside the routing element
        const models = routingEl.querySelectorAll('.routing-message-model');
        let currentIndex = 0;
        this.routingAnimationActive = true;
        const animateModels = () => {
            if (!this.routingAnimationActive) {
                return;
            }
            models.forEach(m => m.classList.remove('active', 'fading'));
            models[currentIndex].classList.add('active');
            const prevIndex = (currentIndex - 1 + models.length) % models.length;
            const nextIndex = (currentIndex + 1) % models.length;
            models[prevIndex].classList.add('fading');
            models[nextIndex].classList.add('fading');
            currentIndex = (currentIndex + 1) % models.length;
            if (this.routingAnimationActive) {
                setTimeout(animateModels, 300);
            }
        };
        animateModels();
    }
            models.forEach(m => m.classList.remove('active', 'fading'));
            models[currentIndex].classList.add('active');
            const prevIndex = (currentIndex - 1 + models.length) % models.length;
            const nextIndex = (currentIndex + 1) % models.length;
            models[prevIndex].classList.add('fading');
            models[nextIndex].classList.add('fading');
            currentIndex = (currentIndex + 1) % models.length;
            if (this.routingAnimationActive) {
                setTimeout(animateModels, 300);
            }
        };
        animateModels();
    }

    hideRoutingAnimation(force = false) {
        if (force) {
            // Immediate removal, ignore minimum display time
            this.routingAnimationActive = false;
            if (this.routingMessageElement) {
                // Hide the routing element instead of removing it
                this.routingMessageElement.style.display = 'none';
                this.routingMessageElement.innerHTML = '';
                // Keep reference for future use
                // this.routingMessageElement = null;
            }
            return;
        }
        const elapsed = Date.now() - this.routingAnimationStartTime;
        const remaining = Math.max(0, this.minRoutingAnimationDuration - elapsed);

        setTimeout(() => {
            this.routingAnimationActive = false;
            // Clear placeholder content and hide it
            const routingEl = document.getElementById('routingMessage');
            if (routingEl) {
                routingEl.innerHTML = '';
                routingEl.style.display = 'none';
            }
        }, remaining);
    }

    showTyping() {
        this.typingElement = document.createElement('div');
        this.typingElement.className = 'message regolo-brick';
        this.typingElement.id = 'typingIndicator';
        this.typingElement.innerHTML = `
            <div class="message-header">
                <span class="sender">regolo-brick</span>
            </div>
            <div class="typing">
                <span class="typing-dots">
                    <span></span><span></span><span></span>
                </span>
            </div>
        `;
        this.elements.messages.appendChild(this.typingElement);
        this.scrollToBottom();
    }

    hideTyping() {
        if (this.typingElement) {
            this.typingElement.remove();
            this.typingElement = null;
        }
    }

    scrollToBottom() {
        this.elements.messages.scrollTop = this.elements.messages.scrollHeight;
    }

    clearChat() {
        this.messages = [];
        this.elements.messages.innerHTML = `
            <div class="empty-state">
                <p>Chat cleared. Start a new conversation!</p>
            </div>
        `;
        localStorage.removeItem(STORAGE_KEY);
    }

    saveHistory() {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(this.messages));
        } catch (e) {
            console.warn('Could not save chat history:', e);
        }
    }

    loadHistory() {
        try {
            const saved = localStorage.getItem(STORAGE_KEY);
            if (saved) {
                const history = JSON.parse(saved);
                if (history.length > 0) {
                    this.elements.messages.innerHTML = '';
                    history.forEach(msg => {
                        this.messages.push(msg);
                        this.renderMessage(msg);
                    });
                }
            }
        } catch (e) {
            console.warn('Could not load chat history:', e);
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML.replace(/\n/g, '<br>');
    }
}

const app = new ChatApp();
