
    
    <script>
	
        let currentFile = '';  // Store the currently opened file name

        // Handle drag and drop for the entire messages area
        const messages = document.getElementById('messages');

        messages.addEventListener('dragover', (e) => {
            e.preventDefault();
            messages.classList.add('dragover');
        });

        messages.addEventListener('dragleave', () => {
            messages.classList.remove('dragover');
        });

        messages.addEventListener('drop', (e) => {
            e.preventDefault();
            messages.classList.remove('dragover');

            const items = e.dataTransfer.items;
            if (items) {
                handleFileUpload(items);
            }
        });
		
let currentSortMethod = 'name';  // Track the current sort method

// Fetch and display the list of files, with toggled sort direction
function fetchFileList(sortMethod = 'name') {
    // Check if the same sort method was selected, and toggle the direction
    const isSameMethod = (sortMethod === currentSortMethod);
    currentSortMethod = sortMethod;  // Update the current method

    fetch(`/list-files?sort=${sortMethod}`)
        .then(response => response.json())
        .then(data => {
            const fileList = document.getElementById('file-list');
            fileList.innerHTML = '';  // Clear the list

            data.files.forEach(file => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.className = 'file-checkbox';
                checkbox.value = file;

                fileItem.appendChild(checkbox);
                fileItem.appendChild(document.createTextNode(` ${file}`));
                fileItem.onclick = () => openFile(file);

                fileList.appendChild(fileItem);
            });
        })
        .catch(error => {
            console.error('Error fetching file list:', error);
            document.getElementById('file-list').textContent = 'Failed to load files.';
        });
}

// Attach event listener to the sort dropdown
document.getElementById('sort-method').addEventListener('change', (e) => {
    fetchFileList(e.target.value);
});

// Open a file in the editor
function openFile(filename) {
    currentFile = filename;

    fetch(`/read-file/${filename}`)
        .then(response => response.text())
        .then(content => {
            document.getElementById('editor').value = content;
        })
        .catch(error => console.error('Error reading file:', error));
}

// Save changes made in the editor
function saveFile() {
    const content = document.getElementById('editor').value;

    fetch('/save-file', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: currentFile, content })
    })
        .then(response => response.json())
        .then(result => alert(result.message))
        .catch(error => console.error('Error saving file:', error));
}

// Run the script in a virtual environment or emulator
function runScript() {
    fetch(`/run-script/${currentFile}`)
        .then(response => response.json())
        .then(result => alert(`Script Output:\n${result.output}`))
        .catch(error => console.error('Error running script:', error));
}

// Delete selected files
function deleteSelectedFiles() {
    const checkboxes = document.querySelectorAll('.file-checkbox:checked');
    const filesToDelete = Array.from(checkboxes).map(cb => cb.value);

    if (filesToDelete.length === 0) {
        alert('Please select at least one file to delete.');
        return;
    }

    fetch('/delete-files', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ files: filesToDelete })
    })
        .then(response => response.json())
        .then(result => {
            alert(result.message);
            fetchFileList();  // Refresh the file list
        })
        .catch(error => console.error('Error deleting files:', error));
}

        // Load the file list when the page is ready
        document.addEventListener('DOMContentLoaded', fetchFileList);
		
		    // Konami Code sequence
    const konamiCode = [
        "ArrowUp", "ArrowUp",
        "ArrowDown", "ArrowDown",
        "ArrowLeft", "ArrowRight",
        "ArrowLeft", "ArrowRight"
    ];
    let konamiIndex = 0;

    // Listen for keydown events
    document.addEventListener('keydown', (event) => {
        const key = event.key;
        
        // Check if the key matches the current step in the sequence
        if (key === konamiCode[konamiIndex]) {
            konamiIndex++;
        } else {
            konamiIndex = 0; // Reset if the sequence breaks
        }

        // If the entire sequence is completed, trigger the disco biscuit mode
        if (konamiIndex === konamiCode.length) {
            triggerDiscoBiscuit();
            konamiIndex = 0; // Reset for potential future use
        }
    });

        // Auto-expand the textarea to fit content
        const userInput = document.getElementById('user-input');
        userInput.addEventListener('input', function () {
            this.style.height = 'auto'; // Reset height to allow shrinking
            this.style.height = `${this.scrollHeight}px`; // Adjust height based on content
        });
		
document.addEventListener('DOMContentLoaded', () => {
    const chatWindow = document.getElementById('messages');
    if (!chatWindow) {
        console.error("Chat window not found.");
        return;
    }

    const processedUrls = new Set(); // Track processed URLs
    const allowedTools = ['bash', 'pip'];
    const toolPattern = new RegExp(`(${allowedTools.join('|')})\\s+(.+)`, 'i');
    const urlPattern = /(https?|ftp|sftp):\/\/[^\s/$.?#].[^\s]*/gi;
    const codePattern = /```(bash|python|perl|ruby|javascript)?([\s\S]*?)```/gi;
    const logScraperPattern = /log_scraper\s+(.+)/i;

    let accumulatedScripts = {}; // Store accumulated scripts by language


    // Unified function to observe and handle all new messages
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            mutation.addedNodes.forEach(node => {
                if (node.nodeType !== Node.ELEMENT_NODE) return;

                const messageContent = node.textContent.trim();

                // Handle code blocks
                let match;
                while ((match = codePattern.exec(messageContent)) !== null) {
                    const language = match[1] || 'unknown';
                    const scriptContent = match[2].trim();
                    console.log(`Detected code block in ${language}:`, scriptContent);

                    accumulateCodeBlock(language, scriptContent);
                }

                // Handle URLs
                const urls = messageContent.match(urlPattern);
                if (urls) urls.forEach(handleGenericUrl);
				
                // Detect and handle log_scraper command
                const logScraperMatch = messageContent.match(logScraperPattern);
                if (logScraperMatch) {
                    const args = logScraperMatch[1].trim();
                    console.log(`Detected log_scraper command with arguments: ${args}`);
                    triggerLogScraper(args);  // Call the API with extracted arguments
                }

                // Handle tool commands
                const toolMatch = messageContent.match(toolPattern);
                if (toolMatch) {
                    const [_, toolName, toolArgument] = toolMatch;
                    console.log(`Detected tool: ${toolName} with argument: ${toolArgument}`);
                    executeToolCommand(toolName, toolArgument);
                }
            });
        });
    });

    observer.observe(chatWindow, { childList: true, subtree: true });

    // Function to accumulate code blocks
    function accumulateCodeBlock(language, content) {
        if (!accumulatedScripts[language]) {
            accumulatedScripts[language] = [];
        }
        accumulatedScripts[language].push(content);
    }

    // Function to trigger the /log-scraper API endpoint
    function triggerLogScraper(args) {
        fetch('/log-scraper', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ arg: args })
        })
        .then(response => response.json())
        .then(result => {
            const message = result.success 
                ? `Log scraper output: ${result.output}` 
                : `Error: ${result.error}`;
            appendToChatWindow('Log Scraper', message);
        })
        .catch(error => {
            appendToChatWindow('Log Scraper', `Error: ${error.message}`);
        });
    }
	
    // Function to save accumulated code blocks
async function saveCodeBlock(language) {
    if (!accumulatedScripts[language]) return;

    const completeScript = accumulatedScripts[language].join('\n');
    
    try {
        // Await filename generation before proceeding
        const filename = await generateFilename(language, completeScript);
        console.log(`Saving complete ${language} script to: ${filename}`);  // Debug log
        
        // Now that the filename is ready, call ScriptSaver
        ScriptSaver(filename, completeScript);
        
        // Clear accumulated scripts after saving
        delete accumulatedScripts[language];
    } catch (error) {
        console.error('Error saving code block:', error);
    }
}

    // Call this function periodically to save all accumulated code blocks
    setInterval(() => {
        for (const language in accumulatedScripts) {
            saveCodeBlock(language);
        }
    }, 10000);

    // Handle URLs
    function handleGenericUrl(url) {
        if (processedUrls.has(url)) return;

        processedUrls.add(url);
        const formData = new FormData();
        formData.append('generic_url', url);
        //appendToChatWindow('URL Observer', 'Auto-embedding...');

        fetch('/embed-content', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            //.then(result => appendToChatWindow('URL Observer', result.result || "Content embedded."))
            //.catch(error => appendToChatWindow('URL Observer', `Error - ${error.message}`));
    }

    // Execute tool commands
    function executeToolCommand(toolName, toolArgument) {
        fetch('/tools/allowed')
            .then(response => response.text())
            .then(text => {
                const allowedCommands = text.split('\n').map(cmd => cmd.trim());
                if (toolName === 'bash' && !allowedCommands.includes(toolArgument.split(' ')[0])) {
                    //appendToChatWindow('Tool Observer', `Command "${toolArgument}" is not allowed.`);
                    return;
                }

                fetch(`/tools/${toolName}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ argument: toolArgument })
                })
                    .then(response => response.json())
                    .then(result => appendToChatWindow('Tool Observer', formatTextAsHtml(result.result)))
                    //.catch(error => appendToChatWindow('Tool Observer', `Error: ${error.message}`));
            })
            .catch(error => appendToChatWindow('Tool Observer', `Error loading allowed commands: ${error.message}`));
    }
});
	
    let discoInterval;
	

    function createSparkle() {
        const sparkle = document.createElement('div');
        sparkle.className = 'sparkle';
        sparkle.style.left = `${Math.random() * 100}vw`;
        sparkle.style.top = `${Math.random() * 100}vh`;
        const size = Math.random() * 10 + 5;
        sparkle.style.width = `${size}px`;
        sparkle.style.height = `${size}px`;
        document.body.appendChild(sparkle);

        setTimeout(() => {
            sparkle.style.opacity = '0';
            setTimeout(() => {
                sparkle.remove();
            }, 1000);
        }, Math.random() * 2000 + 500);
    }


    function startDiscoBiscuitBackground() {
        if (discoInterval) return;

        discoInterval = setInterval(() => {
            document.body.style.background = `hsl(${Math.floor(Math.random() * 360)}, 100%, 50%)`;

            for (let i = 0; i < 5; i++) {
                createSparkle();
            }
        }, 500);
    }

    function stopDiscoBiscuitBackground() {
        clearInterval(discoInterval);
        discoInterval = null;
        document.body.style.background = '';
        document.querySelectorAll('.sparkle').forEach(sparkle => sparkle.remove());
    }

    function triggerDiscoBiscuit() {
        fetch('/disco_biscuit', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.message.includes("activated")) {
                startDiscoBiscuitBackground();
            } else {
                console.log(data.message || 'Disco Biscuit mode already active!');
            }
        })
        .catch(error => {
            console.error('Error triggering Disco Biscuit:', error);
        });
    }

    // Poll for Disco Biscuit status every 10 seconds
    setInterval(() => {
        fetch('/disco_biscuit_status')
            .then(response => response.json())
            .then(data => {
                if (!data.active && discoInterval) {
                    stopDiscoBiscuitBackground();
                }
            })
            .catch(error => console.error('Error checking Disco Biscuit status:', error));
    }, 100000); // Check every 10 seconds
	


		
		function analyzeImage() {
    const formData = new FormData();
    const file = document.getElementById('image').files[0];
    const model = 'Avery'; // Set Avery as the model for image analysis.

    if (!file) {
        alert("Please select an image to analyze.");
        return;
    }

    formData.append('file', file);
    formData.append('model', model);

    fetch('/analyze-image', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
        } else if (data.result) {
            // Insert the result as a message from Avery
            appendToChatWindow("Avery", formatTextAsHtml(data.result));

            // Allow AI1, AI2, or Gabriel to respond to Avery's message
            handleBotReplyToAvery(data.result);
        } else {
            alert("Error: No result returned from server.");
        }
    })
    .catch(error => {
        console.error('Error analyzing image:', error);
    });
}



function handleBotReplyToAvery(message) {
    // Randomly select a bot to reply to Avery
    const botModels = ['AI1', 'AI2', 'Gabriel'];
    const replyingBot = botModels[Math.floor(Math.random() * botModels.length)];
    
    const num_ctx = parseInt(document.getElementById('num_ctx').value) || 2048;
    const modelAI = document.getElementById(replyingBot.toLowerCase() + 'Model').value || replyingBot;
            const maxTokens = parseInt(document.getElementById('max-tokens-input').value) || -2;

    // Check if the replying bot is muted
    if (document.getElementById(replyingBot.toLowerCase() + 'MuteToggle').checked) {
        return;
    }

    // Let the selected bot reply to Avery's message
    fetch('/ollama', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            model: modelAI,
            prompt: message,
            search_embeddings: document.getElementById('searchEmbeddingsToggle')?.checked ? "yes" : "no",
            stream: false,
					options: {
                        num_predict: maxTokens,
                        num_ctx: num_ctx,
						temperature: 1
						},
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.response) {
            // Insert the bot's response into the chatbox
            appendToChatWindow(modelAI, formatTextAsHtml(data.response));
        }
    })
    .catch(error => {
        console.error('Error in bot reply:', error);
    });
}

    function handleFileUpload(items) {
        const modelAI1 = getCookie('AI1') || 'Alex';
        const messageDiv = document.getElementById('messages');

        for (let i = 0; i < items.length; i++) {
            let entry = items[i].webkitGetAsEntry ? items[i].webkitGetAsEntry() : items[i].getAsEntry ? items[i].getAsEntry() : null;

            if (entry) {
                if (entry.isDirectory) {
                    traverseDirectory(entry);
                } else if (entry.isFile) {
                    entry.file(file => uploadFile(file, modelAI1));
                }
            } else if (items[i] instanceof File) {
                uploadFile(items[i], modelAI1);
            }
        }

        function traverseDirectory(directory) {
            const reader = directory.createReader();
            reader.readEntries(entries => {
                if (entries.length === 0) {
                    console.warn('No files found in directory.');
                }
                for (let j = 0; j < entries.length; j++) {
                    if (entries[j].isFile) {
                        entries[j].file(file => uploadFile(file, modelAI1));
                    } else if (entries[j].isDirectory) {
                        traverseDirectory(entries[j]);
                    }
                }
            }, error => {
                console.error('Error reading directory:', error);
                const errorMessage = document.createElement('p');
                errorMessage.textContent = `Error reading directory: ${error.message}`;
                messageDiv.appendChild(errorMessage);
            });
        }
    }
	
    // Helper function to upload a single file
    function uploadFile(file, model, url = null) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model', model);

        if (url) {
            formData.append('generic_url', url);
        }

        fetch('/upload-log', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
        if (data.error && data.needs_credentials) {
            // If authentication is needed, prompt the user for credentials
            promptForCredentials(url, model);
        } else if (data.summaries) {
            // Handle successful response with summaries
            data.summaries.forEach(summaryData => {
                displaySummary(summaryData);
            });
			
        } else if (data.error) {
            displayError(data.error);
        }
    })
    .catch(error => {
        console.error('Error uploading file:', error);
        displayError(`Error uploading file: ${error.message}`);
    });
}

function promptForCredentials(url, model) {
    const username = prompt("Enter your username for accessing the URL:");
    if (username) {
        const password = prompt("Enter your password:");
        if (password) {
            // Retry uploading with the provided credentials
            const formData = new FormData();
            formData.append('generic_url', url);
            formData.append('model', model);
            formData.append('username', username);
            formData.append('password', password);

            fetch('/embed-content', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.result) {
                    // Successfully embedded the content
                    const successMessage = document.createElement('p');
                    successMessage.textContent = `username observer: ${data.result}`;
                    document.getElementById('messages').appendChild(successMessage);
                } else if (data.error) {
                    displayError(data.error);
                }
            })
            .catch(error => {
                console.error('Error retrying with credentials:', error);
                displayError(`Error retrying with credentials: ${error.message}`);
            });
        }
    }
}


// Function to display summaries
function displaySummary(summaryData) {
    const { model, file, part, summary } = summaryData;
    const summaryMessage = document.createElement('div');
    summaryMessage.innerHTML = `
        <strong>Model:</strong> ${model}<br>
        <strong>File:</strong> ${file}<br>
        <strong>Part:</strong> ${part}<br>
        <strong>Summary:</strong> ${formatTextAsHtml(summary)}
    `;
    document.getElementById('messages').appendChild(summaryMessage);
}

// Function to display error messages
function displayError(errorMessage) {
    const errorDiv = document.createElement('p');
    errorDiv.textContent = `Error: ${errorMessage}`;
    document.getElementById('messages').appendChild(errorDiv);
    document.getElementById('messages').scrollTop = document.getElementById('messages').scrollHeight;
}


// Function to format text into HTML, preserving newlines and code blocks with backticks
function formatTextAsHtml(text) {
    // Preserve triple backticks as-is to detect code blocks later
    let formattedText = text.replace(/```([\s\S]*?)```/g, '```$1```');

    // Preserve inline backticks (single line code)
    formattedText = formattedText.replace(/`([^`]+)`/g, '`$1`');

    // Replace remaining newlines with <br> for line breaks
    formattedText = formattedText.replace(/\n/g, '<br>');

    // Handle numbered lists by adding line breaks before each number
    formattedText = formattedText.replace(/(\d+\.)\s/g, '<br>$1 ');

    // Format bold text using <strong> tags for star lists
    formattedText = formattedText.replace(/\*\*\s*(.*?)\*\*/g, '<strong>$1</strong>');

    // Format bullet points with <li> tags
    formattedText = formattedText.replace(/^\s*[-*]\s+(.*)/gm, '<li>$1</li>');

    return formattedText;
}

        function getCookie(name) {
            const nameEQ = name + "=";
            const ca = document.cookie.split(';');
            for (let i = 0; i < ca.length; i++) {
                let c = ca[i];
                while (c.charAt(0) === ' ') c = c.substring(1);
                if (c.indexOf(nameEQ) === 0) return c.substring(nameEQ.length, c.length);
            }
            return null;
		}
       
function appendToChatWindow(sender, message) {
    const chatWindow = document.getElementById('messages');  // Ensure chatWindow is defined within the function
    if (!chatWindow) {
        console.error("Chat window not found.");
        return;
    }

    const newMessage = document.createElement('div');
    newMessage.innerHTML = `<strong>${sender}:</strong> ${message}<br>`;
    chatWindow.appendChild(newMessage);
    chatWindow.scrollTop = chatWindow.scrollHeight;  // Auto-scroll to the bottom
}

// Generate filename with appropriate extension using Gabriel
async function generateFilename(language, completeScript) {
    //const extension = languageExtensions[language.toLowerCase()] || 'txt';  // Default to .txt if unknown

    // Send script to Gabriel for context-based naming
    const context_name = await nameMe(completeScript);  // Await the response
    return `${context_name}`;
}


// Modified nameMe function to get context-based filename from Gabriel
function nameMe(script) {
    return fetch('/ollama', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            model: 'Gabriel',
            prompt: `Please read the following script and suggest a contextually appropriate filename: (respond with only the filename) \n\n${script}`,
            stream: false,
            options: { num_predict: 10, num_ctx: 4096 }
        })
    })
    .then(response => response.json())
    .then(data => data.response.trim())  // Ensure proper trimming
    .catch(error => {
        console.error('Error fetching filename from Gabriel:', error);
        return 'default_name';  // Fallback if Gabriel fails
    });
}

// ScriptSaver function to save the filename and content
function ScriptSaver(filename, content) {
    fetch('/script-saver', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, content })
    })
    .then(response => response.json())
    //.then(result => appendToChatWindow('ScriptSaver', result.success 
    //    ? `${result.message}` 
    //    : `Error: ${result.message}`))
    .catch(error => appendToChatWindow('ScriptSaver', `Error: ${error.message}`));
}

    // Function to Handle USER submitting Messages from USER ONLY
    async function sendMessage() {
        const messageDiv = document.getElementById('messages');
        const message = document.getElementById('user-input').value.trim();
		console.log("User message:", message); 

        if (!message) return; // Exit if message is empty

        // Display User Message
            appendToChatWindow('You', formatTextAsHtml(message));
            userInput.value = ''; // Clear input
            userInput.style.height = 'auto'; // Reset height
            handleBotResponse(message);
	

        // Handle URLs (GitHub, Confluence, Generic, etc.)
        const githubPattern = /https:\/\/git\.[\w.-]+\/[\w.-]+\/[\w.-]+/;
        const confluencePattern = /https:\/\/[\w.-]+\/(?:pages\/viewpage\.action\?spaceKey=[\w-]+&title=[\w-]+|display\/[\w-]+\/[\w+-]+)/;
        const orgPattern = /https:\/\/git\.[\w.-]+\/orgs\/[\w-]+\/repositories/;
        const sshPattern = /[\w-]+@[\w.-]+:[\w-]+\/[\w.-]+\.git/;
        const publicGithubPattern = /https:\/\/github\.com\/[\w-]+\/[\w.-]+\.git/;
        const genericUrlPattern = /^(https:\/\/(?!git\.|github\.com|confluence\.|.*confluence.*)[\w.-]+(?:\/[\w.-]*)*)$/;

        if (githubPattern.test(message) || confluencePattern.test(message) || orgPattern.test(message) || sshPattern.test(message) || publicGithubPattern.test(message) || genericUrlPattern.test(message)) {
            const formData = new FormData();

            // Handle GitHub HTTPS URL
            if (githubPattern.test(message)) {
                formData.append('github', message.match(githubPattern)[0]);
            }

            // Handle GitHub SSH URL
            if (sshPattern.test(message)) {
                formData.append('ssh_url', message.match(sshPattern)[0]);
            }

            // Handle Public GitHub URL
            if (publicGithubPattern.test(message)) {
                formData.append('public_github', message.match(publicGithubPattern)[0]);
            }

            // Handle Confluence URL
            if (confluencePattern.test(message)) {
                const confluenceMatch = message.match(confluencePattern);
                formData.append('confluence', confluenceMatch[0]);

                // Extract space key and title if available
                const viewPageMatch = confluenceMatch[0].match(/spaceKey=([\w-]+)&title=([\w-]+)/);
                const displayMatch = confluenceMatch[0].match(/display\/([\w-]+)\/([\w+-]+)/);
                if (viewPageMatch) {
                    formData.append('space_key', viewPageMatch[1]);
                    formData.append('title', viewPageMatch[2].replace('+', ' '));
                } else if (displayMatch) {
                    formData.append('space_key', displayMatch[1]);
                    formData.append('title', displayMatch[2].replace('+', ' '));
                }
            }

            // Handle Generic URLs (non-reserved)
            if (genericUrlPattern.test(message)) {
                const genericUrl = message.match(genericUrlPattern)[0];
                formData.append('generic_url', genericUrl);
            }

            // Handle GitHub Organization URL
            if (orgPattern.test(message)) {
                const orgUrl = message.match(orgPattern)[0];
                formData.append('org_url', orgUrl);
            }

            messageDiv.innerHTML += `<p>Handle URLs observer: Detected a URL. Auto-submitting for embedding...</p>`;

            fetch('/embed-content', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(result => {
                messageDiv.innerHTML += `<p>Handle URLs observer: ${result.result || "Content successfully embedded."}</p>`;
            })
            .catch(error => {
                messageDiv.innerHTML += `<p>Handle URLs observer: Error - ${error.message}</p>`;
            });

            messageDiv.scrollTop = messageDiv.scrollHeight;
        } else {        
            // Handle regular chat conversation with the bot
            // handleBotResponse(message);
        }
    }
	
// Function to submit messages to the chat for *AI1. 
	// All messages are always sent to Gabriel.
	// Gabriel can be muted. He will still listen, but wont respond.
	// All questions are always sent to *AI2.
	// SEND FULL RESPONSE sends all messages to *AI2.

function handleBotResponse(message) {
    const modelAI1 = getCookie('AI1') || 'Alex';
    const modelAI2 = getCookie('AI2') || 'Gemma'; // Add definition for modelAI2
    const gabrielModel = getCookie('gabrielModel') || 'Gabriel';
    const num_ctx = parseInt(document.getElementById('num_ctx').value) || 2048;
    const messageDiv = document.getElementById('messages'); // Make sure this element exists
    const isAtBottom = messageDiv.scrollTop + messageDiv.clientHeight >= messageDiv.scrollHeight - 10;
            const maxTokens = parseInt(document.getElementById('max-tokens-input').value) || -2;
			
		

    // Check if AI1 is muted
    if (!document.getElementById('ai1MuteToggle').checked) {
        fetch('/ollama', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: modelAI1,
                prompt: message,
                search_embeddings: document.getElementById('searchEmbeddingsToggle')?.checked ? "yes" : "no",
                stream: false,
				options: {
                    num_predict: maxTokens,
					temperature: 1,
                    num_ctx: num_ctx
					},
            }),
        })
        .then(response => response.json())
        .then(data => {
            let responseAI1 = data.response;
            appendToChatWindow(modelAI1, formatTextAsHtml(responseAI1));  // Use appendToChatWindow to add the response 

            if (document.getElementById('contextToggle').checked) {
                sendTomodelAI2(responseAI1, modelAI2); // Send full response to AI2
            } else {
                    // Extract questions or tasks for further processing
                    const questions = extractQuestions(responseAI1);
                    if (questions.length > 0) {
                        const combinedQuestions = questions.join(' ');
                        sendToOtherBots(combinedQuestions, modelAI2, gabrielModel);
                        questions.forEach(question => {
                            responseAI1 = responseAI1.replace(question, '').trim();
                        });
                }
            }

                // Optionally speak the response
                if (responseAI1) speak(responseAI1, modelAI1);

                // Forward response to Gabriel if not muted
                if (!document.getElementById('gabrielMuteToggle').checked) {
                    sendToGabriel(`${modelAI1}: ${responseAI1}`);
                }

                // Scroll to the bottom of the chat window if needed
                if (isAtBottom) messageDiv.scrollTop = messageDiv.scrollHeight;

                // Clear the input field after sending
                document.getElementById('user-input').value = '';
            })
            .catch(error => appendToChatWindow('handleBotResponse', `Error: ${error.message}`));
    }
}  //  sample script saved to tools/

// Map of language to file extensions
const languageExtensions = {
    python: 'py',
    bash: 'sh',
    javascript: 'js',
    ruby: 'rb',
    perl: 'pl',
    java: 'java',
    go: 'go',
    c: 'c',
    cpp: 'cpp',
    rust: 'rs',
};


	

            const maxTokens = parseInt(document.getElementById('max-tokens-input').value) || -2;


    document.getElementById('user-input').addEventListener('keydown', function (event) {
        if (event.key === 'Enter') {
            sendMessage();
        }
    });


        function sendToGabriel(conversation) {
		model = 'Gabriel'
            fetch('/ollama', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: model,
                    prompt: conversation,
					search_embeddings: document.getElementById('searchEmbeddingsToggle')?.checked ? "yes" : "no",
                    stream: false,
					options: {
                        num_predict: maxTokens,
                        num_ctx: 4096
						},
                }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.response && !document.getElementById('gabrielMuteToggle').checked) {
                    //const messageDiv = document.getElementById('messages');
                    //const gabrielMessage = document.createElement('p');
                    //gabrielMessage.textContent = `${data.response}`;
                    //messageDiv.appendChild(gabrielMessage);
					
					appendToChatWindow(model, formatTextAsHtml(data.response)); // Bold name
					
                }
            })
            .catch(error => {
                appendToChatWindow('Error', `${error.message}`);
            });
        }


// Send Alex's response to other bots (Gemma and Gabriel)
function sendToOtherBots(message, modelAI2, gabrielModel) {
    // Send to Gemma (modelAI2) if not muted
    if (!document.getElementById('ai2MuteToggle').checked) {
        sendTomodelAI(message, modelAI2);
    }

    // Always send to Gabriel (no mute toggle for demonstration purposes)
    sendToGabriel(message);
}

// Send message to a specific AI (e.g., Gemma)
function sendTomodelAI(message, model) {
    const messageDiv = document.getElementById('messages');
    fetch('/ollama', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model, prompt: message, stream: false }),
    })
	
        .then(response => response.json())
        .then(data => {
            appendToChatWindow(model, formatTextAsHtml(data.response));
			handleBotResponse(data.response);
			sendTomodelAI2(data.response, model);
            messageDiv.scrollTop = messageDiv.scrollHeight;
        })
        .catch(error => appendToChatWindow('sendTomodelAI', `Error: ${error.message}`));
}

function sendTomodelAI2(message, model) {
            // Check if AI2 is muted
            if (document.getElementById('ai2MuteToggle').checked) {
                return;
            }

            const messageDiv = document.getElementById('messages');
            const maxTokens = parseInt(document.getElementById('max-tokens-input').value) || -2;
            const num_ctx = parseInt(document.getElementById('num_ctx').value) || 2048;

            fetch('/ollama', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: model,  // Use selected model for AI2
                    prompt: message,
					search_embeddings: document.getElementById('searchEmbeddingsToggle')?.checked ? "yes" : "no",
                    stream: false,
					options: {
                        num_predict: maxTokens,
                        num_ctx: num_ctx,
						temperature: 1
						},
                }),
            })
            .then(response => response.json())
            .then(data => {
				appendToChatWindow(model, formatTextAsHtml(data.response)); // Bold Gemma's name
		    	handleBotResponse(data.response);
                speak(data.response, model);

                messageDiv.scrollTop = messageDiv.scrollHeight;
            })
            .catch(error => {
                const errorMessage = document.createElement('p');
                errorMessage.textContent = `Error: ${error}`;
                messageDiv.appendChild(errorMessage);
            });
        }

        function updateModelCookie(model) {
            const modelValue = document.getElementById(model === 'AI1' ? 'ai1Model' : 'ai2Model').value;
            setCookie(model, modelValue, 365);
        }


        function extractQuestions(text) {
            const questionRegex = /[^.?!]*(\?+)/g;
            const matches = text.match(questionRegex);
            return matches ? matches.map(match => match.trim()) : [];
        }

        function speak(text, model) {
            const synth = window.speechSynthesis;
            const utterThis = new SpeechSynthesisUtterance(text);
            const voices = synth.getVoices();
            if (model === 'Aqua') {
                utterThis.voice = voices.find(voice => voice.name.includes("Google UK English Male")) || voices[0];
                utterThis.pitch = 1.0;
            } else if (model === 'Gemma') {
                utterThis.voice = voices.find(voice => voice.name.includes("Google UK English Female")) || voices[1];
                utterThis.pitch = 1.2;
            }
            synth.speak(utterThis);
        }
		
function saveConversation() {
    const messageDiv = document.getElementById('messages');
    let conversationText = '';

    // Loop through all child nodes of the messages div, regardless of type
    messageDiv.childNodes.forEach(node => {
        // Check if the node has text content and add it to the conversation text
        if (node.textContent.trim()) {
            conversationText += `${node.textContent.trim()}\n`;
        }
    });

    // Create a Blob and download the conversation as a text file
    const blob = new Blob([conversationText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'conversation.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

        function setCookie(name, value, days) {
            const d = new Date();
            d.setTime(d.getTime() + (days * 24 * 60 * 60 * 1000));
            const expires = "expires=" + d.toUTCString();
            document.cookie = name + "=" + value + ";" + expires + ";path=/";
        }

        document.addEventListener('DOMContentLoaded', function () {
            const savedModelAI1 = getCookie('AI1') || 'Alex';
            const savedModelAI2 = getCookie('AI2') || 'Gemma';
            document.getElementById('ai1Model').value = savedModelAI1;
            document.getElementById('ai2Model').value = savedModelAI2;
        });
		
function getLastTwoMessages() {
    const messageContainer = document.getElementById('messages');

    if (!messageContainer) {
        console.error('Chat window (#messages) not found.');
        return {
            original_request: '',
            task_description: 'Chat window not found.'
        };
    }

    // Querying all messages that have <strong> elements (usernames) and div content
    const messages = Array.from(messageContainer.querySelectorAll('div'));
    const filteredMessages = messages.filter(msg => msg.textContent.trim().length > 0);

    const messageCount = filteredMessages.length;

    if (messageCount < 2) {
        console.warn('Not enough messages to extract.');
        return {
            original_request: '',
            task_description: 'No sufficient messages available.'
        };
    }

    // Extracting the two most recent messages
    const taskDescription = filteredMessages[messageCount - 1].textContent.trim();
    const originalRequest = filteredMessages[messageCount - 2].textContent.trim();

    return {
        original_request: originalRequest,
        task_description: taskDescription
    };
}


// Add the workflow trigger logic
function triggerJAMbotWorkflow() {
    const { original_request, task_description } = getLastTwoMessages();

    fetch('/trigger-workflow', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            original_request: original_request,
            task_description: task_description
        })
    })
    .then(response => response.json())
    .then(data => {
        appendToChatWindow('Workflow', `${data.response}`);
    })
    .catch(error => {
        appendToChatWindow('Workflow', `Error triggering workflow: ${error.message}`);
    });
}

// Attach event listener to the workflow button
document.getElementById('workflowButton').addEventListener('click', function () {
    triggerJAMbotWorkflow();
});
