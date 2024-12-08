// EventsAdapter.java
package com.atakmap.android.soundstrike.plugin;


import android.content.Context;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;
import com.atakmap.android.soundstrike.plugin.R;


import java.util.List;

public class EventsAdapter extends BaseAdapter {
    private Context context;
    private List<GunshotEvent> events;

    public EventsAdapter(Context context, List<GunshotEvent> events) {
        this.context = context;
        this.events = events;
    }

    @Override
    public int getCount() {
        return events.size();
    }

    @Override
    public Object getItem(int position) {
        return events.get(position);
    }

    @Override
    public long getItemId(int position) {
        return position;
    }

    // ViewHolder pattern for performance optimization
    static class ViewHolder {
        TextView eventTimeView;
        TextView eventDetailsView;
    }

    @Override
    public View getView(int position, View convertView, ViewGroup parent) {
        ViewHolder holder;
        if (convertView == null) {
            convertView = LayoutInflater.from(context).inflate(R.layout.event_row, parent, false);
            holder = new ViewHolder();
            holder.eventTimeView = convertView.findViewById(R.id.event_time);
            holder.eventDetailsView = convertView.findViewById(R.id.event_details);
            convertView.setTag(holder);
        } else {
            holder = (ViewHolder) convertView.getTag();
        }

        GunshotEvent event = events.get(position);

        holder.eventTimeView.setText("Time: " + event.eventTime);
        String details = String.format("Lat: %.4f, Lon: %.4f, Caliber: %s, Conf: %.2f%%",
                event.latitude, event.longitude, event.caliber, event.confidence);
        holder.eventDetailsView.setText(details);

        return convertView;
    }
}
