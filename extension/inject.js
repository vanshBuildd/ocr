(function() {
    var script = document.createElement('script');
    script.src = chrome.runtime.getURL('my_script.js');
    (document.head || document.documentElement).appendChild(script);
})();