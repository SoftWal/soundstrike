// GunshotEvent.java
package com.atakmap.android.soundstrike.plugin;


public class GunshotEvent {
    public String eventTime;
    public double latitude;
    public double longitude;
    public String caliber;
    public double confidence;

    public GunshotEvent(String eventTime, double latitude, double longitude, String caliber, double confidence) {
        this.eventTime = eventTime;
        this.latitude = latitude;
        this.longitude = longitude;
        this.caliber = caliber;
        this.confidence = confidence;
    }
}
