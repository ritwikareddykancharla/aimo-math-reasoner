$packages = @(
    "xifthen",
    "microtype",
    "hyperref",
    "psnfss",
    "xcolor",
    "textcase",
    "colortbl",
    "booktabs",
    "changepage",
    "enumitem",
    "datetime",
    "fancyhdr",
    "lastpage",
    "titlesec",
    "bibentry",
    "mdframed",
    "caption",
    "needspace",
    "XCharter",
    "newtx",
    "zlmtt",
    "geometry",
    "authblk",
    "natbib",
    "multirow",
    "tablefootnote",
    "listings",
    "tcolorbox",
    "soul",
    "iftex",
    "etoolbox",
    "algorithms",
    "algorithmicx"
)

foreach ($pkg in $packages) {
    Write-Host "Installing $pkg..."
    mpm --install $pkg
}
