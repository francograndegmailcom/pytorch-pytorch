import requests


class HTTPIO():
    def __init__(self, url) -> None:
        self.cur = 0
        self.url = url
        # use HTTP keep-alive
        self.session = requests.Session()
        # use HTTP HEAD to fetch file size
        self.size = int(requests.head(self.url).headers['Content-Length'])

    def seek(self, offset: int, whence=0) -> int:
        if whence == 0:
            self.cur = offset
        elif whence == 1:
            self.cur += offset
        else:
            self.cur = self.size + offset

        return self.cur

    def tell(self) -> int:
        return self.cur

    def read(self, size: int = -1, /) -> bytes | None:
        if size < 0:
            r = requests.get(self.url)
            return bytes(r.text)

        range = 'bytes=%d-%d' % (self.cur, self.cur + size - 1)
        headers = {'Range': range}
        r = requests.get(self.url, headers=headers)
        self.cur += size
        return r.content

    def readline(self, size: int | None = -1, /) -> bytes:
        if size < 0:
            line = self._readline()
        else:
            line = self.read(size)

        return line

    def _readline(self):
        # line terminator is always b'\n' for binary files
        line = bytearray()
        while True:
            char = self.read(1)
            if char == b'\n':
                line += b'\n'
                break

            if char == b'':
                break

            line += char

        return line
    
    def close(self):
        self.session.close()
