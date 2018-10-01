package com.example.SimpleLayout;

import android.app.ActionBar;
import android.app.Activity;
import android.content.ContentValues;
import android.content.Intent;
import android.os.Bundle;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

public class Register extends Activity {

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.register_menu, menu);
        ActionBar actionBar = getActionBar();
        actionBar.setDisplayShowTitleEnabled(true);
        actionBar.show();
        return true;
    }

    /**
     * Called when the activity is first created.
     */
    Button register;
    EditText email;
    EditText password1;
    EditText password2;
    TextView messages;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.register);
        register = (Button)findViewById(R.id.registerButton);
        email = (EditText)findViewById(R.id.emailEdit);
        password1 = (EditText)findViewById(R.id.password1Edit);
        password2 = (EditText)findViewById(R.id.password2Edit);
        messages = (TextView)findViewById(R.id.messagesText);

        register.setEnabled(false);
        email.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {}
            @Override
            public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {}
            @Override
            public void afterTextChanged(Editable editable) {
                EnableRegistration(email.getText().toString(),password1.getText().toString(),password2.getText().toString());
            }
        });

        password1.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {}
            @Override
            public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {}
            @Override
            public void afterTextChanged(Editable editable) {
                EnableRegistration(email.getText().toString(),password1.getText().toString(),password2.getText().toString());
            }
        });

        password2.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {}
            @Override
            public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {}
            @Override
            public void afterTextChanged(Editable editable) {
                EnableRegistration(email.getText().toString(),password1.getText().toString(),password2.getText().toString());
            }
        });

        register.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                String user_password = password1.getText().toString();
                String user_email = email.getText().toString();

                TextView remind = (TextView) findViewById(R.id.messagesText);
                ContentValues params = new ContentValues();
                params.put("email",user_email);
                params.put("password",user_password);
                AsyncHTTPRequest AsyncRegister = new AsyncHTTPRequest(
                        "http://lamp.ms.wits.ac.za/~s1312548/register.php",params) {
                    @Override
                    protected void onPostExecute(String output) {
                        if (output.equals("mellon"))
                        {
                            Intent intent = new Intent(Register.this, main.class);
                            intent.putExtra("email", user_email);
                            Register.this.startActivity(intent);
                        }
                        else
                        {
                            remind.setText("It looks like that username is taken");
                        }

                    }
                };
                AsyncRegister.execute();
            }
        });
    }

    private String EncryptPassword(String password){
        String encryptedPassword = "";

        return encryptedPassword;
    }

    private boolean PasswordsMatch(String password1, String password2){
        if(password1.equals(password2)){
            return true;
        }else{
            return false;
        }
    }

    private void EnableRegistration(String email, String password1, String password2){
        String passwordMessage = "", emailMessage = "";

        boolean emailEntered = false, passwordsEntered = false;
        if(email.equals("")){
            emailEntered = false;
            emailMessage = "E-mail address not entered";
        }else{
            emailEntered = true;
        }
        if(password1.equals("") || password2.equals("")){
            passwordsEntered = false;
            passwordMessage = "Empty password field(s)";
        }else{
            passwordsEntered = true;
        }

        if(PasswordsMatch(password1,password2)){

        }else{
           passwordMessage = "Passwords do not match";
        }

        if(passwordsEntered && PasswordsMatch(password1,password2) && emailEntered){
            messages.setText(emailMessage+"\n"+passwordMessage+"\n");
            register.setEnabled(true);
        }else{
            messages.setText(emailMessage+"\n"+passwordMessage+"\n");
            register.setEnabled(false);
        }
    }

    public void get_help(MenuItem item) {
        String url = "http://lamp.ms.wits.ac.za/~s1312548/help.html";
        Intent myIntent = new Intent(Register.this, WebStuuf.class);
        myIntent.putExtra("key", url);
        Register.this.startActivity(myIntent);
    }
}
