chrome.action.onClicked.addListener((tab) => {
    chrome.scripting.executeScript({
        target: {tabId: tab.id},
        function: function() {
            // Your script content here
            console.log('This is my injected script!');
        }
    });
});
