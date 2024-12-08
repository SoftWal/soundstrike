// SoundStrike.java
package com.atakmap.android.soundstrike.plugin;


import android.content.Context;
import android.graphics.Point;
import android.graphics.Color;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ListView;
import android.widget.AdapterView;
import android.widget.TextView;
import android.widget.Toast;

import com.atak.plugins.impl.PluginContextProvider;
import com.atak.plugins.impl.PluginLayoutInflater;
import com.atakmap.android.soundstrike.plugin.R;
import gov.tak.api.plugin.IPlugin;
import gov.tak.api.plugin.IServiceController;
import gov.tak.api.ui.IHostUIService;
import gov.tak.api.ui.Pane;
import gov.tak.api.ui.PaneBuilder;
import gov.tak.api.ui.ToolbarItem;
import gov.tak.api.ui.ToolbarItemAdapter;
import gov.tak.platform.marshal.MarshalManager;

import org.json.JSONArray;
import org.json.JSONException;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;

public class SoundStrike implements IPlugin {

    private IServiceController serviceController;
    private Context pluginContext;
    private IHostUIService uiService;
    private ToolbarItem toolbarItem;
    private Pane templatePane;

    private ListView eventsListView;
    private EventsAdapter eventsAdapter;
    private List<GunshotEvent> eventsList = new ArrayList<>();

    // Marker option pane variables
    private Pane markerPane;
    private double pendingLat;
    private double pendingLon;
    private String pendingLabel;

    // Custom Marker View
    private CustomMarkerView customMarkerView;

    public SoundStrike(IServiceController serviceController) {
        this.serviceController = serviceController;
        final PluginContextProvider ctxProvider = serviceController.getService(PluginContextProvider.class);
        if (ctxProvider != null) {
            pluginContext = ctxProvider.getPluginContext();
            pluginContext.setTheme(R.style.ATAKPluginTheme);
        }

        uiService = serviceController.getService(IHostUIService.class);

        toolbarItem = new ToolbarItem.Builder(
                pluginContext.getString(R.string.app_name),
                MarshalManager.marshal(
                        pluginContext.getResources().getDrawable(R.drawable.ic_launcher),
                        android.graphics.drawable.Drawable.class,
                        gov.tak.api.commons.graphics.Bitmap.class))
                .setListener(new ToolbarItemAdapter() {
                    @Override
                    public void onClick(ToolbarItem item) {
                        showPane();
                    }
                })
                .build();
    }

    @Override
    public void onStart() {
        if (uiService == null) return;
        uiService.addToolbarItem(toolbarItem);

    }

    @Override
    public void onStop() {
        if (uiService == null) return;
        uiService.removeToolbarItem(toolbarItem);
    }

    private void showPane() {
        if (templatePane == null) {
            View rootView = PluginLayoutInflater.inflate(pluginContext, R.layout.main_layout, null);
            eventsListView = rootView.findViewById(R.id.events_list);
            Button refreshButton = rootView.findViewById(R.id.refresh_button);

            // Initialize CustomMarkerView and add it to the rootView
            customMarkerView = new CustomMarkerView(pluginContext);
            // Set layout parameters to match the parent
            customMarkerView.setLayoutParams(new ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT));
            // Make the CustomMarkerView transparent and overlaying
            customMarkerView.setBackgroundColor(Color.TRANSPARENT);
            // Add the CustomMarkerView to the rootView
            if (rootView instanceof ViewGroup) {
                ((ViewGroup) rootView).addView(customMarkerView);
            }

            templatePane = new PaneBuilder(rootView)
                    .setMetaValue(Pane.RELATIVE_LOCATION, Pane.Location.Default)
                    .setMetaValue(Pane.PREFERRED_WIDTH_RATIO, 0.5D)
                    .setMetaValue(Pane.PREFERRED_HEIGHT_RATIO, 0.5D)
                    .build();

            eventsAdapter = new EventsAdapter(pluginContext, eventsList);
            eventsListView.setAdapter(eventsAdapter);

            // On refresh, fetch new data
            refreshButton.setOnClickListener(v -> fetchDatabaseContent());

            // On event click, show marker option pane
            eventsListView.setOnItemClickListener((AdapterView<?> parent, View view, int position, long id) -> {
                GunshotEvent selectedEvent = eventsList.get(position);
                showMarkerOptionPane(selectedEvent.latitude, selectedEvent.longitude, selectedEvent.eventTime);
            });
        }

        if (!uiService.isPaneVisible(templatePane)) {
            uiService.showPane(templatePane, null);
            fetchDatabaseContent();
        }
    }

    private void fetchDatabaseContent() {
        new Thread(() -> {
            String result = fetchDataFromServer("http://172.20.10.6:5000/get_events");
            if (result != null) {
                List<GunshotEvent> loadedEvents = parseEventsJson(result);

                new Handler(Looper.getMainLooper()).post(() -> {
                    eventsList.clear();
                    eventsList.addAll(loadedEvents);
                    eventsAdapter.notifyDataSetChanged();
                });
            } else {
                new Handler(Looper.getMainLooper()).post(() -> {
                    eventsList.clear();
                    eventsAdapter.notifyDataSetChanged();
                    Toast.makeText(pluginContext, "Failed to fetch events.", Toast.LENGTH_SHORT).show();
                });
            }
        }).start();
    }

    private List<GunshotEvent> parseEventsJson(String json) {
        List<GunshotEvent> parsedEvents = new ArrayList<>();
        try {
            JSONArray arr = new JSONArray(json);
            for (int i = 0; i < arr.length(); i++) {
                org.json.JSONObject obj = arr.getJSONObject(i);
                String eventTime = obj.getString("event_time");
                double latitude = obj.getDouble("latitude");
                double longitude = obj.getDouble("longitude");
                String caliber = obj.getString("caliber");
                double confidence = obj.getDouble("confidence");
                parsedEvents.add(new GunshotEvent(eventTime, latitude, longitude, caliber, confidence));
            }
        } catch (JSONException e) {
            e.printStackTrace();
        }
        return parsedEvents;
    }

    private String fetchDataFromServer(String urlString) {
        HttpURLConnection urlConnection = null;
        BufferedReader reader = null;
        try {
            URL url = new URL(urlString);
            urlConnection = (HttpURLConnection) url.openConnection();
            urlConnection.setRequestMethod("GET");
            urlConnection.setConnectTimeout(5000); // 5 seconds timeout
            urlConnection.setReadTimeout(5000); // 5 seconds timeout
            urlConnection.connect();

            int responseCode = urlConnection.getResponseCode();
            System.out.println("Response Code: " + responseCode);
            if (responseCode == 200) {
                reader = new BufferedReader(new InputStreamReader(urlConnection.getInputStream()));
                StringBuilder sb = new StringBuilder();
                String line;
                while ((line = reader.readLine()) != null) {
                    sb.append(line);
                }
                return sb.toString();
            } else {
                // Handle non-200 responses
                return null;
            }
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        } finally {
            if (reader != null) {
                try { reader.close(); } catch (Exception ignored) {}
            }
            if (urlConnection != null) {
                urlConnection.disconnect();
            }
        }
    }

    private void showMarkerOptionPane(double latitude, double longitude, String label) {
        this.pendingLat = latitude;
        this.pendingLon = longitude;
        this.pendingLabel = label;

        View rootView = PluginLayoutInflater.inflate(pluginContext, R.layout.marker_option_layer, null);

        TextView latView = rootView.findViewById(R.id.marker_lat);
        TextView lonView = rootView.findViewById(R.id.marker_lon);
        Button confirmBtn = rootView.findViewById(R.id.confirm_marker_btn);

        latView.setText("Lat: " + latitude);
        lonView.setText("Lon: " + longitude);


        markerPane = new PaneBuilder(rootView)
                .setMetaValue(Pane.RELATIVE_LOCATION, Pane.Location.Default)
                .setMetaValue(Pane.PREFERRED_WIDTH_RATIO, 0.4D)
                .setMetaValue(Pane.PREFERRED_HEIGHT_RATIO, 0.3D)
                .build();

        if (!uiService.isPaneVisible(markerPane)) {
            uiService.showPane(markerPane, null);
        }
    }


}
