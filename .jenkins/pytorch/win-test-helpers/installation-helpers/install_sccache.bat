mkdir %TMP_DIR_WIN%\bin

if "%REBUILD%"=="" (
  :check_sccache
  %TMP_DIR_WIN%\bin\sccache.exe --show-stats || (
    taskkill /im sccache.exe /f /t || ver > nul
    del %TMP_DIR_WIN%\bin\sccache.exe || ver > nul
    del %TMP_DIR_WIN%\bin\sccache-cl.exe || ver > nul
    curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/sccache-20211130.exe --output %TMP_DIR_WIN%\bin\sccache.exe
    curl --retry 3 -k https://s3.amazonaws.com/ossci-windows/sccache-cl.exe --output %TMP_DIR_WIN%\bin\sccache-cl.exe
    goto :check_sccache
  )
)
