// Verify logic separated to ensure independence

async function verifyGraph() {
    // Relying on global `window.lastGraphData` which is populated by app.js when processing completes.
    
    if (typeof window.lastGraphData === 'undefined' || !window.lastGraphData || !window.lastGraphData.nodes) {
        alert("No graph data available to verify. Run reasoning first.");
        return;
    }
    
    const verifyBtn = document.getElementById('verify-btn');
    const originalHTML = verifyBtn.innerHTML;
    verifyBtn.innerHTML = 'Verifying...';
    verifyBtn.disabled = true;
    
    // Adapter at the edge: Map app's flat nodes/edges into [{"id", "text", "parent_ids"}]
    const mappedNodes = window.lastGraphData.nodes.map(n => {
        const pIds = (window.lastGraphData.edges || [])
            .filter(e => e.target === n.id)
            .map(e => String(e.source));
        return { id: String(n.id), text: n.text, parent_ids: pIds };
    });
    
    try {
        const response = await fetch('/api/verify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ nodes: mappedNodes })
        });
        
        const data = await response.json();
        
        if (data.success) {
            applyVerificationFlags(data.flags);
        } else {
            console.error('Verification failed:', data.error);
            alert("Verification failed: " + data.error);
        }
    } catch (e) {
        console.error('Error verifying graph:', e);
    } finally {
        verifyBtn.innerHTML = originalHTML;
        verifyBtn.disabled = false;
    }
}

function applyVerificationFlags(flags) {
    if (!flags || Object.keys(flags).length === 0) {
        alert("Verification passed: No arithmetic or logic errors found!");
        return;
    }
    
    const container = document.getElementById('verification-results');
    if (container) {
        container.innerHTML = '';
        container.style.display = 'block';
    }
    
    let styleStr = '\n\n%% Verification Styling\n';
    let htmlContent = '<h4>Verification Issues</h4><ul style="margin:0; padding-left:20px; font-size:13px; color:#dc2626;">';
    
    for (const [nodeId, nodeFlags] of Object.entries(flags)) {
        styleStr += `style ${nodeId} fill:#fecaca,stroke:#dc2626\n`;
        
        nodeFlags.forEach(f => {
            const confidenceStr = f.source === 'nli' ? ` (conf: ${f.confidence.toFixed(2)})` : '';
            htmlContent += `<li style="margin-bottom:5px;">
                <span style="display:inline-block; padding:2px 6px; background:#dc2626; color:white; border-radius:4px; font-size:10px; margin-right:5px; text-transform:uppercase;">${f.type}</span>
                <b>Node ${nodeId}:</b> ${f.reason}${confidenceStr}
            </li>`;
        });
    }
    htmlContent += '</ul>';
    
    if (container) {
        container.innerHTML = htmlContent;
    }
    
    // Inject the style into the Mermaid code and re-render
    if (typeof window.lastMermaidCode !== 'undefined' && window.lastMermaidCode) {
        // Only append if not already appended
        let newCode = window.lastMermaidCode;
        if (!newCode.includes('%% Verification Styling')) {
            newCode += styleStr;
        }
        
        // Use the app's global render function if available
        if (typeof mermaid !== 'undefined') {
            const diagramContainer = document.getElementById('mermaid-diagram');
            diagramContainer.removeAttribute('data-processed');
            diagramContainer.innerHTML = `<div class="mermaid">${newCode}</div>`;
            mermaid.init(undefined, document.querySelectorAll('.mermaid'));
        }
    }
}
