mkdir %TMP_DIR_WIN%\bin

if "%REBUILD%"=="" (
  :check_sccache
  %TMP_DIR_WIN%\bin\sccache.exe --show-stats || (
    taskkill /im sccache.exe /f /t || ver > nul
    del %TMP_DIR_WIN%\bin\sccache.exe || ver > nul
    if "%BUILD_ENVIRONMENT%"=="" (
      curl --retry 3 --retry-all-errors -k https://s3.amazonaws.com/ossci-windows/sccache-v0.7.4.exe --output %TMP_DIR_WIN%\bin\sccache.exe
    ) else (
      aws s3 cp s3://ossci-windows/sccache-v0.7.4.exe %TMP_DIR_WIN%\bin\sccache.exe
    )
    goto :check_sccache
  )
)