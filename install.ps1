param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$scriptDir/install.py" @ScriptArgs
exit $LASTEXITCODE
