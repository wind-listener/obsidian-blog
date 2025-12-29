class HomepageMasonry {
    constructor() {
        this.container = document.getElementById('posts-masonry');
        this.currentMode = 'recent';
        this.loadedPosts = new Set();
        this.postsPerPage = 50;
        this.currentPage = 0;
        this.allPosts = [];

        this.init();
    }

    async init() {
        // Load all posts data
        await this.loadPostsData();

        // Setup mode switcher
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchMode(e.target.dataset.mode);
            });
        });

        // Setup infinite scroll
        this.setupInfiniteScroll();

        // Initial load
        await this.loadPosts();

        // Initial layout
        this.relayoutMasonry();

        // Relayout on resize
        let resizeTimer;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimer);
            resizeTimer = setTimeout(() => this.relayoutMasonry(), 250);
        });
    }

    async loadPostsData() {
        const response = await fetch('/index.json');
        const data = await response.json();

        // For now, all posts have 0 backlinks (would need Hugo custom output)
        this.allPosts = data.map(post => ({
            ...post,
            backlinkCount: post.backlinks || 0
        }));
    }

    switchMode(mode) {
        this.currentMode = mode;
        this.currentPage = 0;
        this.loadedPosts.clear();
        this.container.innerHTML = '';

        // Update active button
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });

        this.loadPosts().then(() => this.relayoutMasonry());
    }

    getFilteredPosts() {
        let posts = [...this.allPosts];

        switch(this.currentMode) {
            case 'recent':
                return posts.sort((a, b) => new Date(b.date) - new Date(a.date));
            case 'backlinks':
                return posts.sort((a, b) => b.backlinkCount - a.backlinkCount);
            case 'random':
                return posts.sort(() => Math.random() - 0.5);
            default:
                return posts;
        }
    }

    async loadPosts() {
        const filtered = this.getFilteredPosts();
        const start = this.currentPage * this.postsPerPage;
        const end = start + this.postsPerPage;
        const posts = filtered.slice(start, end);

        if (posts.length === 0) {
            document.getElementById('load-more-btn').style.display = 'none';
            return;
        }

        posts.forEach(post => {
            if (!this.loadedPosts.has(post.url)) {
                this.container.insertAdjacentHTML('beforeend', this.createPostCard(post));
                this.loadedPosts.add(post.url);
            }
        });

        this.currentPage++;

        // Show/hide load more button
        const hasMore = (this.currentPage * this.postsPerPage) < filtered.length;
        document.getElementById('load-more-btn').style.display = hasMore ? 'block' : 'none';
    }

    createPostCard(post) {
        const date = new Date(post.date).toLocaleDateString('zh-CN');
        const section = post.section || '';
        const backlinks = post.backlinkCount ? `<span class="backlinks">🔗 ${post.backlinkCount}</span>` : '';
        const summary = post.summary ? post.summary.substring(0, 120) + '...' : '';

        return `
            <a href="${post.url}" class="post-card">
                <h3>${post.title}</h3>
                <div class="post-meta">
                    <time>${date}</time>
                    ${backlinks}
                    ${section ? `<span class="section-name">${section}</span>` : ''}
                </div>
                ${summary ? `<p class="post-summary">${summary}</p>` : ''}
            </a>
        `;
    }

    setupInfiniteScroll() {
        const observer = new IntersectionObserver((entries) => {
            if (entries[0].isIntersecting) {
                this.loadPosts().then(() => this.relayoutMasonry());
            }
        }, { threshold: 0.1 });

        observer.observe(document.getElementById('load-more-btn'));
    }

    relayoutMasonry() {
        const cards = Array.from(this.container.querySelectorAll('.post-card'));
        if (cards.length === 0) return;

        // Determine column count based on screen width
        let columnCount;
        if (window.innerWidth >= 1600) {
            columnCount = 4;
        } else if (window.innerWidth >= 1200) {
            columnCount = 3;
        } else if (window.innerWidth >= 768) {
            columnCount = 2;
        } else {
            columnCount = 1;
        }

        const gap = 32;
        const cardWidth = (this.container.offsetWidth - (gap * (columnCount - 1))) / columnCount;
        const columnHeights = new Array(columnCount).fill(0);

        cards.forEach((card) => {
            const shortestColumn = columnHeights.indexOf(Math.min(...columnHeights));
            const left = shortestColumn * (cardWidth + gap);
            const top = columnHeights[shortestColumn];

            card.style.left = `${left}px`;
            card.style.top = `${top}px`;
            card.style.width = `${cardWidth}px`;
            card.style.position = 'absolute';

            columnHeights[shortestColumn] += card.offsetHeight + gap;
        });

        this.container.style.height = `${Math.max(...columnHeights)}px`;
    }
}

// Initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => new HomepageMasonry());
} else {
    new HomepageMasonry();
}