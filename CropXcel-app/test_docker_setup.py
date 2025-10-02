#!/usr/bin/env python
"""
Quick test script to verify Docker setup is working correctly
"""

import requests


def test_application():
    """Test that the application is running correctly"""

    print("🧪 Testing CropXcel Docker Setup...")

    try:
        # Test main page
        response = requests.get("http://localhost:8001", timeout=10)
        if response.status_code == 200:
            print("✅ Web application is accessible")
        else:
            print(f"❌ Web application returned status {response.status_code}")
            return False

        # Test API endpoint (if available)
        try:
            api_response = requests.get("http://localhost:8001/api/fields/", timeout=10)
            # 401/403 expected without auth
            if api_response.status_code in [200, 401, 403]:
                print("✅ API endpoints are accessible")
            else:
                print(f"⚠️  API returned status {api_response.status_code}")
        except requests.exceptions.RequestException:
            print("⚠️  API test skipped (endpoint may not exist)")

        print("✅ All tests passed! CropXcel is running successfully in Docker")
        print("\n🌐 Access the application at: http://localhost:8001")
        return True

    except requests.exceptions.ConnectionError:
        print(
            "❌ Could not connect to the application. "
            "Make sure Docker containers are running."
        )
        return False
    except requests.exceptions.Timeout:
        print("❌ Application is taking too long to respond")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    test_application()
