import io
import unittest

from app import app


class UploadApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_upload_requires_exactly_four_videos(self):
        data = {"videos": [(io.BytesIO(b"fake"), "a.mp4")]}
        response = self.client.post('/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)

    def test_status_unknown_job(self):
        response = self.client.get('/status/does-not-exist')
        self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()
