import asyncio
import ssl
import os
import logging
import requests

# Enable logging for debugging
logging.basicConfig(level=logging.DEBUG)


# Function to get machine's approximate coordinates
def get_coordinates():
    try:
        response = requests.get("http://ip-api.com/json/")
        data = response.json()

        if data["status"] == "success":
            latitude = data["lat"]
            longitude = data["lon"]
            print(f"Latitude: {latitude}, Longitude: {longitude}")
            return latitude, longitude
        else:
            print("Unable to retrieve location.")
            return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None


# Asynchronous function to send message to TAK Server
async def send_message(latitude, longitude):
    tak_server = "192.168.0.32"
    tak_port = 8089  # TLS-encrypted TCP port

    # Construct message with dynamic coordinates
    message = f"""<?xml version='1.0' standalone='yes'?>
    <event version="2.0" uid="SoundResponder" type="a-f-G-U-C" how="m-g">
        <point lat="{latitude}" lon="{longitude}" hae="0.0" ce="9999999.0" le="9999999.0"/>
        <detail>
            <contact callsign="SoundResponder"/>
            <remarks>Test</remarks>
        </detail>
    </event>"""

    # Paths to converted PEM files
    client_cert = os.path.expanduser("~/Coms-Cert/wintak_cert.pem")
    client_key = os.path.expanduser("~/Coms-Cert/wintak_key.pem")
    ca_cert = os.path.expanduser("~/Coms-Cert/TAK-Sound.pem")

    # Verify that the certificate files exist
    assert os.path.exists(client_cert), f"Client cert not found: {client_cert}"
    assert os.path.exists(client_key), f"Client key not found: {client_key}"
    assert os.path.exists(ca_cert), f"CA cert not found: {ca_cert}"

    # Create an SSL context
    ssl_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_cert)
    ssl_context.load_cert_chain(certfile=client_cert, keyfile=client_key)
    ssl_context.check_hostname = False  # Disable hostname checking if necessary

    try:
        # Establish a TCP connection with SSL/TLS
        reader, writer = await asyncio.open_connection(
            tak_server, tak_port, ssl=ssl_context
        )

        # Send the message
        writer.write(message.encode())
        await writer.drain()  # Ensure the message is sent

        # Attempt to read response (if any) from the server
        response = await reader.read(4096)
        if response:
            print("Server response:", response.decode())
        else:
            print("No response from the server.")

        print("Message sent successfully.")

    except ssl.SSLError as ssl_err:
        logging.exception("SSL error occurred while sending the message.")
    except OSError as os_err:
        logging.exception("OS error occurred, could be network-related.")
    except Exception as e:
        logging.exception("An unexpected error occurred.")
    finally:
        writer.close()


# Get coordinates and send the message
latitude, longitude = get_coordinates()
if latitude and longitude:
    asyncio.run(send_message(latitude, longitude))
else:
    print("Could not retrieve coordinates, message not sent.")
