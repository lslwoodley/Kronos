// Kronos Bot Web UI interactions

document.addEventListener('DOMContentLoaded', () => {
    // Auto-hide mobile menu on navigation
    const mobileMenu = document.getElementById('mobile-menu');
    if (mobileMenu) {
        mobileMenu.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => mobileMenu.classList.add('hidden'));
        });
    }
});

// Helper for HTMX fetch feedback
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `fixed bottom-4 right-4 px-4 py-2 rounded-md text-sm text-white shadow-lg ${type === 'error' ? 'bg-red-600' : 'bg-brand-600'}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

document.body.addEventListener('htmx:responseError', (e) => {
    showToast('Request failed: ' + (e.detail?.xhr?.responseText || 'unknown'), 'error');
});
