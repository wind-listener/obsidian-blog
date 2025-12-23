// 搜索功能
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
let fuse;

// 加载搜索索引
fetch('/index.json')
    .then(response => response.json())
    .then(data => {
        const options = {
            keys: ['title', 'content', 'tags'],
            threshold: 0.3,
            includeScore: true,
            minMatchCharLength: 2
        };
        fuse = new Fuse(data, options);
    })
    .catch(error => console.error('Error loading search index:', error));

// 搜索功能
searchInput.addEventListener('input', function(e) {
    const query = e.target.value.trim();

    if (query.length < 2) {
        searchResults.innerHTML = '';
        return;
    }

    if (!fuse) {
        searchResults.innerHTML = '<p class="no-results">搜索索引加载中...</p>';
        return;
    }

    const results = fuse.search(query);

    if (results.length === 0) {
        searchResults.innerHTML = '<p class="no-results">没有找到相关结果</p>';
        return;
    }

    const htmlParts = results.slice(0, 10).map(result => {
        const item = result.item;
        const excerpt = getExcerpt(item.content, query);
        const tags = item.tags ? item.tags.map(tag => '<span class="tag">#' + tag + '</span>').join(' ') : '';

        return '<div class="search-result-item">' +
            '<h3><a href="' + item.url + '">' + item.title + '</a></h3>' +
            '<p class="excerpt">' + excerpt + '</p>' +
            (tags ? '<div class="tags">' + tags + '</div>' : '') +
            '</div>';
    });

    searchResults.innerHTML = htmlParts.join('');
});

function getExcerpt(content, query) {
    const lowerContent = content.toLowerCase();
    const lowerQuery = query.toLowerCase();
    const index = lowerContent.indexOf(lowerQuery);

    if (index === -1) {
        return content.substring(0, 200) + '...';
    }

    const start = Math.max(0, index - 60);
    const end = Math.min(content.length, index + query.length + 140);
    const excerpt = content.substring(start, end);

    return (start > 0 ? '...' : '') + excerpt + (end < content.length ? '...' : '');
}
