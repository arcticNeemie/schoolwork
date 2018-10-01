package com.example.SimpleLayout;

import android.app.Activity;
import android.content.ContentValues;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.util.ArrayList;

/**
 * Created by owere on 2017/05/21.
 */
public class AddFeeds extends Activity {

    Button save;
    CheckBox myTagBox, newTagBox;
    LinearLayout myFeedssLL, newFeedsLL;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.add_feeds);
        Context me = this;
        save = (Button)findViewById(R.id.saveChangesButton);
        myFeedssLL = (LinearLayout)findViewById(R.id.myTagsLinLayout);
        newFeedsLL = (LinearLayout)findViewById(R.id.newTagsLinLayout);

        String user_email = getIntent().getStringExtra("email");
        ContentValues params = new ContentValues();
        params.put("email",user_email);
        ArrayList<String> tags = new ArrayList<>();
        AsyncHTTPRequest AsyncGetMyTags = new AsyncHTTPRequest(
                "http://lamp.ms.wits.ac.za/~s1312548/get_my_feeds.php",params) {
            @Override
            protected void onPostExecute(String output) {
                try{
                    JSONArray myTags = new JSONArray(output);
                    if(myTags.length()==0){
                        TextView noTags = new TextView(AddFeeds.this);
                        noTags.setText("Seems like you have not selected any feeds");
                        myFeedssLL.addView(noTags);
                    }else{
                        for (int i = 0; i < myTags.length(); i++){
                            JSONObject item=myTags.getJSONObject(i);
                            String tag_name = item.getString("FEED_NAME");
                            myTagBox = new CheckBox(AddFeeds.this);
                            myTagBox.setText(tag_name);
                            myTagBox.setChecked(true);
                            myFeedssLL.addView(myTagBox);
                            tags.add(tag_name);
                        }
                    }
                }catch (Exception e){}

            }
        };
        AsyncGetMyTags.execute();

        AsyncHTTPRequest AsyncGetOtherTags = new AsyncHTTPRequest(
                "http://lamp.ms.wits.ac.za/~s1312548/get_feeds.php",params) {
            @Override
            protected void onPostExecute(String output) {
                try{
                    JSONArray otherTags = new JSONArray(output);
                    if(otherTags.length()==0){
                        TextView noTags = new TextView(AddFeeds.this);
                        noTags.setText("No new feeds available");
                        newFeedsLL.addView(noTags);
                    }else{
                        for (int i = 0; i < otherTags.length(); i++){
                            JSONObject item=otherTags.getJSONObject(i);
                            String tag_name = item.getString("FEED_NAME");
                            if (!tags.contains(tag_name))
                            {
                                newTagBox = new CheckBox(AddFeeds.this);
                                newTagBox.setText(tag_name);
                                newTagBox.setChecked(false);
                                newFeedsLL.addView(newTagBox);
                            }

                        }
                    }
                }catch (Exception e){}

            }
        };
        AsyncGetOtherTags.execute();




        save.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                for(int i = 0; i < newFeedsLL.getChildCount(); i++){
                    View chkView =  newFeedsLL.getChildAt(i);
                    if (chkView instanceof CheckBox)
                    {
                        CheckBox box = (CheckBox) chkView;
                        if(box.isChecked()){
                            String tagName = box.getText().toString();
                            ContentValues params = new ContentValues();
                            params.put("email",user_email);
                            params.put("feed",tagName);
                            AsyncHTTPRequest AsyncAddTag = new AsyncHTTPRequest(
                                    "http://lamp.ms.wits.ac.za/~s1312548/add_userfeeds.php",params) {
                                @Override
                                protected void onPostExecute(String output) {
                                    Log.d("FEEDS",output);
                                }
                            };
                            AsyncAddTag.execute();
                        }
                    }


                }

                for(int i = 0; i < myFeedssLL.getChildCount(); i++){
                    View chkView = myFeedssLL.getChildAt(i);
                    if (chkView instanceof CheckBox)
                    {
                        CheckBox box = (CheckBox) chkView;
                        if(!box.isChecked()){
                            String tagName = box.getText().toString();
                            ContentValues params = new ContentValues();
                            params.put("email",user_email);
                            params.put("feed",tagName);
                            AsyncHTTPRequest AsyncDeleteTag = new AsyncHTTPRequest(
                                    "http://lamp.ms.wits.ac.za/~s1312548/remove_userfeeds.php",params) {
                                @Override
                                protected void onPostExecute(String output) {
                                    Log.d("FEEDS",output);
                                }
                            };
                            AsyncDeleteTag.execute();

                        }
                    }

                }
                finish();
            }
        });

    }
}