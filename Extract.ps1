
$content = Get-Content -Raw "src/pages/ResearchPage.jsx"
$s = $content.IndexOf("{activeTab === 'gnn'")
$e = $content.IndexOf("{activeTab === 'scanner'")
$e2 = $content.IndexOf("{activeTab === 'pipeline'")
Write-Output "Start: $s, Next: $e, Pipeline: $e2"

