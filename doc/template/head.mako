<style>
.highlight-line {
    background: #ffa;
}
</style>

<script>
// Slightly hacky - we want this to run before highlight-js
document.addEventListener("DOMContentLoaded", function() {
    // Look for the magic sequence "#<==" and highlight what comes before it
    const pattern = /(^\s*)(.+?)\s*#&lt;==/gm;
    const replacement = "$1<mark class=\"highlight-line\">$2</mark>";
    for (let code of document.getElementsByTagName("code")) {
        code.innerHTML = code.innerHTML.replaceAll(pattern, replacement);
    }
});
</script>
