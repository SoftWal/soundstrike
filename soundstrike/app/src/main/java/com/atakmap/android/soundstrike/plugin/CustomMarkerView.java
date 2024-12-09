// CustomMarkerView.java
package com.atakmap.android.soundstrike.plugin;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.view.View;

import java.util.ArrayList;
import java.util.List;

public class CustomMarkerView extends View {
    private List<Marker> markers;
    private Paint paint;
    private Paint textPaint;

    public CustomMarkerView(Context context) {
        super(context);
        markers = new ArrayList<>();
        paint = new Paint();
        paint.setColor(Color.RED);
        paint.setStyle(Paint.Style.FILL);
        paint.setAntiAlias(true);

        textPaint = new Paint();
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(20);
        textPaint.setAntiAlias(true);
    }

    // Method to add a marker
    public void addMarker(int x, int y, String label) {
        markers.add(new Marker(x, y, label));
        invalidate(); // Redraw the view
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        for (Marker marker : markers) {
            // Draw circle
            canvas.drawCircle(marker.x, marker.y, 20, paint);
            // Draw label
            canvas.drawText(marker.label, marker.x + 25, marker.y + 5, textPaint);
        }
    }

    // Inner class to represent a marker
    private static class Marker {
        int x;
        int y;
        String label;

        Marker(int x, int y, String label) {
            this.x = x;
            this.y = y;
            this.label = label;
        }
    }
}
